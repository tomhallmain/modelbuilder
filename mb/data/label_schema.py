"""
Multi-label dataset contract: an ordered label list plus a per-image sidecar manifest.

The folder a file sits in stays its *primary* label, so every step that operates per class
folder — balancing, the test split, the review and rejection flows — keeps working
unchanged. The manifest only adds the labels a folder cannot express:

    data/
      train/<primary_label>/<hash>.jpg
      test/<primary_label>/<hash>.jpg
      label_schema.json      ordered labels, axis grouping, default thresholds
      labels.jsonl           one record per image that carries more than its folder label

An existing single-label dataset is therefore already a valid multi-label dataset with an
empty manifest, which is what allows labels to be added incrementally rather than in one
flag-day re-annotation.

Label order comes from the schema, never from directory sort order. Output index order has
to survive adding a label later: appending to ``labels`` must not permute the indices a
shipped model was trained against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

LABEL_SCHEMA_FILENAME = "label_schema.json"
LABEL_MANIFEST_FILENAME = "labels.jsonl"

SCHEMA_VERSION = 1

# Used when a label has no entry in ``default_thresholds``.
FALLBACK_THRESHOLD = 0.5


class LabelSchemaError(ValueError):
    """Raised when a schema or manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class LabelAxis:
    """
    A named group of labels that vary along one dimension.

    Args:
        name: Axis name, e.g. ``sexual_severity``.
        labels: Member labels, in increasing order when :attr:`ordered`.
        ordered: True when the members form a severity scale rather than unrelated classes.
    """

    name: str
    labels: List[str]
    ordered: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return {"ordered": self.ordered, "labels": list(self.labels)}


@dataclass(frozen=True)
class LabelSchema:
    """Ordered label set for a multi-label dataset."""

    labels: List[str]
    axes: Dict[str, LabelAxis] = field(default_factory=dict)
    default_thresholds: Dict[str, float] = field(default_factory=dict)
    default_threshold: float = FALLBACK_THRESHOLD
    schema_version: int = SCHEMA_VERSION

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def index_of(self, label: str) -> int:
        """Output index for *label*.

        Raises:
            LabelSchemaError: If the label is not in the schema.
        """
        try:
            return self.labels.index(label)
        except ValueError:
            raise LabelSchemaError(
                _("Label {label} is not in the schema.").format(label=label)
            ) from None

    def threshold_for(self, label: str) -> float:
        """Decision threshold for *label*, falling back to :attr:`default_threshold`."""
        return float(self.default_thresholds.get(label, self.default_threshold))

    def thresholds(self) -> List[float]:
        """Per-label thresholds in output-index order."""
        return [self.threshold_for(name) for name in self.labels]

    def multi_hot(self, labels: Iterable[str]) -> List[float]:
        """Multi-hot vector over :attr:`labels` for the given label names."""
        vector = [0.0] * len(self.labels)
        for name in labels:
            vector[self.index_of(name)] = 1.0
        return vector

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "labels": list(self.labels),
            "axes": {name: axis.to_jsonable() for name, axis in self.axes.items()} or None,
            "default_thresholds": dict(self.default_thresholds) or None,
            "default_threshold": self.default_threshold,
        }


def _parse_axes(raw: Any, labels: Sequence[str]) -> Dict[str, LabelAxis]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise LabelSchemaError(_("Schema 'axes' must be an object."))

    known = set(labels)
    axes: Dict[str, LabelAxis] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise LabelSchemaError(
                _("Axis {name} must be an object with a 'labels' list.").format(name=name)
            )
        members = body.get("labels")
        if not isinstance(members, list) or not members:
            raise LabelSchemaError(
                _("Axis {name} must list at least one label.").format(name=name)
            )
        unknown = [m for m in members if m not in known]
        if unknown:
            raise LabelSchemaError(
                _("Axis {name} references labels not in the schema: {unknown}").format(
                    name=name, unknown=", ".join(str(u) for u in unknown)
                )
            )
        axes[str(name)] = LabelAxis(
            name=str(name),
            labels=[str(m) for m in members],
            ordered=bool(body.get("ordered", False)),
        )
    return axes


def parse_label_schema(raw: Mapping[str, Any]) -> LabelSchema:
    """
    Build a :class:`LabelSchema` from decoded JSON.

    Raises:
        LabelSchemaError: If the document is malformed or internally inconsistent.
    """
    labels_raw = raw.get("labels")
    if not isinstance(labels_raw, list) or not labels_raw:
        raise LabelSchemaError(_("Schema must contain a non-empty 'labels' list."))

    labels = [str(name).strip() for name in labels_raw]
    if any(not name for name in labels):
        raise LabelSchemaError(_("Schema labels must not be empty strings."))

    duplicates = sorted({name for name in labels if labels.count(name) > 1})
    if duplicates:
        raise LabelSchemaError(
            _("Schema has duplicate labels: {names}").format(names=", ".join(duplicates))
        )

    thresholds_raw = raw.get("default_thresholds") or {}
    if not isinstance(thresholds_raw, dict):
        raise LabelSchemaError(_("Schema 'default_thresholds' must be an object."))
    unknown = [name for name in thresholds_raw if name not in set(labels)]
    if unknown:
        raise LabelSchemaError(
            _("Thresholds reference labels not in the schema: {unknown}").format(
                unknown=", ".join(str(u) for u in unknown)
            )
        )

    try:
        default_threshold = float(raw.get("default_threshold", FALLBACK_THRESHOLD))
    except (TypeError, ValueError):
        raise LabelSchemaError(_("Schema 'default_threshold' must be a number.")) from None

    return LabelSchema(
        labels=labels,
        axes=_parse_axes(raw.get("axes"), labels),
        default_thresholds={name: float(v) for name, v in thresholds_raw.items()},
        default_threshold=default_threshold,
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
    )


def label_schema_path(data_dir: Path) -> Path:
    """Conventional schema location under a prepared dataset directory."""
    return Path(data_dir) / LABEL_SCHEMA_FILENAME


def label_manifest_path(data_dir: Path) -> Path:
    """Conventional manifest location under a prepared dataset directory."""
    return Path(data_dir) / LABEL_MANIFEST_FILENAME


def load_label_schema(path: Path) -> LabelSchema:
    """
    Load a schema from a file, or from :data:`LABEL_SCHEMA_FILENAME` inside a directory.

    Raises:
        LabelSchemaError: If the file is missing, unreadable, or invalid.
    """
    path = Path(path)
    if path.is_dir():
        path = label_schema_path(path)
    if not path.is_file():
        raise LabelSchemaError(
            _("Label schema not found: {path}").format(path=path)
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise LabelSchemaError(
            _("Could not read label schema {path}: {err}").format(path=path, err=e)
        ) from e
    if not isinstance(raw, dict):
        raise LabelSchemaError(_("Label schema must be a JSON object."))
    return parse_label_schema(raw)


def find_label_schema(data_dir: Path) -> Optional[LabelSchema]:
    """Load the schema under *data_dir*, or None when there is none."""
    path = label_schema_path(data_dir)
    if not path.is_file():
        return None
    return load_label_schema(path)


def load_label_manifest(
    path: Path,
    schema: LabelSchema,
    *,
    strict: bool = True,
) -> Dict[str, List[str]]:
    """
    Load per-image extra labels keyed by dataset-relative POSIX path.

    A missing manifest is not an error: it means every image carries only its folder label,
    which is exactly the state an existing single-label dataset is already in.

    Args:
        path: Manifest file, or a directory containing :data:`LABEL_MANIFEST_FILENAME`.
        schema: Schema the labels must belong to.
        strict: Raise on a label outside the schema. When False, unknown labels are dropped
            with a warning.

    Raises:
        LabelSchemaError: On malformed JSON, a missing ``path`` field, or (when *strict*)
            a label outside the schema.
    """
    path = Path(path)
    if path.is_dir():
        path = label_manifest_path(path)
    if not path.is_file():
        return {}

    known = set(schema.labels)
    out: Dict[str, List[str]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise LabelSchemaError(
            _("Could not read label manifest {path}: {err}").format(path=path, err=e)
        ) from e

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise LabelSchemaError(
                _("Malformed JSON in {path} line {line}: {err}").format(
                    path=path, line=lineno, err=e
                )
            ) from e
        if not isinstance(record, dict):
            raise LabelSchemaError(
                _("Manifest line {line} must be a JSON object.").format(line=lineno)
            )

        rel = str(record.get("path") or "").strip()
        if not rel:
            raise LabelSchemaError(
                _("Manifest line {line} is missing a 'path'.").format(line=lineno)
            )
        rel = rel.replace("\\", "/")

        raw_labels = record.get("labels") or []
        if not isinstance(raw_labels, list):
            raise LabelSchemaError(
                _("Manifest line {line} 'labels' must be a list.").format(line=lineno)
            )

        names: List[str] = []
        for name in raw_labels:
            text_name = str(name).strip()
            if text_name in known:
                names.append(text_name)
            elif strict:
                raise LabelSchemaError(
                    _("Manifest line {line} uses label {label}, which is not in the schema.").format(
                        line=lineno, label=text_name
                    )
                )
            else:
                logger.warning(
                    "Dropping unknown label %r for %s (line %d)", text_name, rel, lineno
                )
        out[rel] = names

    logger.info("Loaded %d manifest entries from %s", len(out), path)
    return out


def report_unresolved_manifest_entries(
    manifest: Mapping[str, Sequence[str]],
    root: Path,
) -> List[str]:
    """
    Manifest keys naming a file that does not exist under *root*, logged as warnings.

    Matching is by exact path only, so an entry written against a path that no longer
    exists contributes nothing — and contributes nothing *silently*, since a missing entry
    is indistinguishable from "this image only has its folder label". The usual cause is
    re-running ``create-dataset`` after annotating: the test split moves files out of
    ``train/`` into ``test/``, and every entry keyed to the old path goes stale.

    Matching those by filename instead was tried and removed: a file with no manifest entry
    can share a basename with one that has an entry, so the fallback assigned labels to
    images that were never annotated. Guessing wrong silently is worse than not matching,
    and the recovery is to re-key the manifest, which this warning prompts.

    Returns:
        The unresolved keys, sorted.
    """
    root = Path(root)
    unresolved = sorted(key for key in manifest if not (root / key).exists())
    if unresolved:
        logger.warning(
            "%d manifest entr%s name files that do not exist under %s; their labels are "
            "not applied. First few: %s",
            len(unresolved),
            "y" if len(unresolved) == 1 else "ies",
            root,
            ", ".join(unresolved[:5]),
        )
    return unresolved


def labels_for_sample(
    relative_path: str,
    primary_label: str,
    manifest: Mapping[str, Sequence[str]],
) -> List[str]:
    """
    Full label set for one image: its folder label plus anything the manifest adds.

    The primary label is always included even if the manifest omits it, so a manifest entry
    can only ever widen an image's labels, never silently remove the one the folder asserts.

    Matching is by exact dataset-relative path. See
    :func:`report_unresolved_manifest_entries` for why there is no filename fallback.
    """
    key = str(relative_path).replace("\\", "/")
    names = [primary_label]
    for name in manifest.get(key, ()):  # type: ignore[arg-type]
        if name != primary_label and name not in names:
            names.append(name)
    return names


def label_positive_counts(
    samples: Iterable[Sequence[float]],
    num_labels: int,
) -> List[int]:
    """Per-label positive counts from multi-hot target vectors."""
    counts = [0] * num_labels
    for vector in samples:
        for index, value in enumerate(vector):
            if index < num_labels and value > 0:
                counts[index] += 1
    return counts


def write_label_schema(path: Path, schema: LabelSchema) -> Path:
    """
    Write *schema* as JSON, creating parent directories.

    *path* is either the dataset directory or the schema file itself, told apart by the
    ``.json`` suffix rather than by :meth:`Path.is_dir`. A write target need not exist yet,
    so an existence check would misread a not-yet-created dataset directory as a filename
    and quietly write the schema to a file named after the directory.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = label_schema_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema.to_jsonable(), indent=2), encoding="utf-8")
    logger.info("Wrote label schema: %s", path)
    return path
