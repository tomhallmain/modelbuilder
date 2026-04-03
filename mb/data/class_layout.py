"""
Class folders under ``raw_data`` / gather sources: discovery and qualifiers.

Pipeline keys (see :mod:`mb.pipeline_config`): ``data.class_names`` (optional list),
``data.class_qualifying_subdir`` (optional str). When the qualifier is set, only
directories containing that immediate child folder count as class (or gather) roots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Default class directory names used by :mod:`tests.fixtures.synthetic_dataset` (legacy three-way split).
SYNTHETIC_DEFAULT_CLASS_NAMES: tuple[str, ...] = ("coherent", "semi-incoherent", "incoherent")

# Post-convert staging under each raw class folder (:mod:`mb.data.convert` writes here).
CONVERTED_MEDIA_SUBDIR = "CONVERTED"
# Legacy name; still recognized for reads and when scanning for conversion inputs.
LEGACY_CONVERTED_MEDIA_SUBDIR = "JPEG_IMAGES"
# Image-classification: one JPEG per video / animated GIF for human review (mirrors CONVERTED output).
VISUAL_MEDIA_REVIEW_SUBDIR = "visual_media_review"

POST_CONVERT_SUBDIR_NAMES: frozenset[str] = frozenset(
    {CONVERTED_MEDIA_SUBDIR, LEGACY_CONVERTED_MEDIA_SUBDIR, VISUAL_MEDIA_REVIEW_SUBDIR}
)

# Immediate children of ``raw_data`` that are pipeline plumbing, not label/staging buckets.
_RAW_DATA_NON_CLASS_SUBDIRS: frozenset[str] = frozenset({"rejected", "small_images_review"})

# Under ``small_images_review``, skip the upscaler output tree when mirroring class folders.
_REVIEW_NON_CATEGORY_SUBDIRS: frozenset[str] = frozenset({"upscaled_small_images"})


def normalize_qualifying_subdir(name: Optional[str]) -> Optional[str]:
    """Return stripped non-empty string or ``None`` (treat empty YAML as unset)."""
    if name is None:
        return None
    s = str(name).strip()
    return s if s else None


def dir_qualifies_as_class(class_dir: Path, class_qualifying_subdir: Optional[str]) -> bool:
    """Whether *class_dir* counts as a class folder for the given qualifier rule."""
    q = normalize_qualifying_subdir(class_qualifying_subdir)
    if not q:
        return True
    return (Path(class_dir) / q).is_dir()


def discover_class_names(
    root: Path,
    *,
    explicit: Optional[Sequence[str]] = None,
    class_qualifying_subdir: Optional[str] = None,
) -> List[str]:
    """
    Resolve class (or label) directory names under *root*.

    If *explicit* is a non-empty sequence, keep only names that exist under *root*,
    satisfy :func:`dir_qualifies_as_class`, and preserve order.

    If *explicit* is ``None`` or empty, list every immediate subdirectory of *root*
    that qualifies (sorted by name).
    """
    root = Path(root)
    q = normalize_qualifying_subdir(class_qualifying_subdir)

    if explicit:
        out: List[str] = []
        for n in explicit:
            name = str(n).strip()
            if not name:
                continue
            p = root / name
            if not p.is_dir():
                continue
            if dir_qualifies_as_class(p, q):
                out.append(name)
        return out

    if not root.is_dir():
        return []

    names: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if dir_qualifies_as_class(p, q):
            names.append(p.name)
    return names


def discover_raw_data_bucket_names(
    raw_data_root: Path,
    *,
    explicit: Optional[Sequence[str]] = None,
    class_qualifying_subdir: Optional[str] = None,
) -> List[str]:
    """
    Staging / class folder names under *raw_data_root* for deduplication and related steps.

    Uses the same rules as :func:`discover_class_names`, then drops known non-class
    directories (``rejected``, ``small_images_review``).
    """
    names = discover_class_names(
        raw_data_root,
        explicit=explicit,
        class_qualifying_subdir=class_qualifying_subdir,
    )
    return [n for n in names if n not in _RAW_DATA_NON_CLASS_SUBDIRS]


def discover_review_bucket_names(
    review_root: Path,
    *,
    explicit: Optional[Sequence[str]] = None,
    class_qualifying_subdir: Optional[str] = None,
) -> List[str]:
    """
    Per-class subfolders under the small-image *review_root* (for upscaling).

    Same discovery as raw buckets, excluding ``upscaled_small_images``.
    """
    names = discover_class_names(
        review_root,
        explicit=explicit,
        class_qualifying_subdir=class_qualifying_subdir,
    )
    return [n for n in names if n not in _REVIEW_NON_CATEGORY_SUBDIRS]


def layout_dict_for_discovery() -> Dict[str, Any]:
    """``class_names`` + ``class_qualifying_subdir`` from the active pipeline config."""
    from mb.pipeline_config import data_class_layout_defaults

    d = data_class_layout_defaults()
    return {
        "explicit": d.get("class_names"),
        "class_qualifying_subdir": d.get("class_qualifying_subdir"),
    }


def resolve_class_media_dir(
    raw_class_dir: Path,
    class_qualifying_subdir: Optional[str],
) -> Optional[Path]:
    """
    Directory that holds source JPEGs for dataset creation (under each class folder).

    If *class_qualifying_subdir* is set, use ``raw_class_dir / <qualifier>`` only if it exists.

    If unset, prefer ``CONVERTED`` (then legacy ``JPEG_IMAGES``), then ``IMAGES``, then fall back to *raw_class_dir*
    (images may live directly in the class folder).
    """
    raw_class_dir = Path(raw_class_dir)
    q = normalize_qualifying_subdir(class_qualifying_subdir)
    if q:
        d = raw_class_dir / q
        return d if d.is_dir() else None
    for name in (CONVERTED_MEDIA_SUBDIR, LEGACY_CONVERTED_MEDIA_SUBDIR, "IMAGES"):
        d = raw_class_dir / name
        if d.is_dir():
            return d
    return raw_class_dir if raw_class_dir.is_dir() else None
