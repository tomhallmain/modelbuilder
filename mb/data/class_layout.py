"""
Class folders under ``raw_data`` / gather sources: discovery and qualifiers.

Pipeline keys (see :mod:`mb.pipeline_config`): ``data.class_names`` (optional list),
``data.class_qualifying_subdir`` (optional str). When the qualifier is set, only
directories containing that immediate child folder count as class (or gather) roots.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

# Default class directory names used by :mod:`tests.fixtures.synthetic_dataset` (legacy three-way split).
SYNTHETIC_DEFAULT_CLASS_NAMES: tuple[str, ...] = ("coherent", "semi-incoherent", "incoherent")


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


def resolve_class_media_dir(
    raw_class_dir: Path,
    class_qualifying_subdir: Optional[str],
) -> Optional[Path]:
    """
    Directory that holds source JPEGs for dataset creation (under each class folder).

    If *class_qualifying_subdir* is set, use ``raw_class_dir / <qualifier>`` only if it exists.

    If unset, prefer ``JPEG_IMAGES``, then ``IMAGES``, then fall back to *raw_class_dir*
    (images may live directly in the class folder).
    """
    raw_class_dir = Path(raw_class_dir)
    q = normalize_qualifying_subdir(class_qualifying_subdir)
    if q:
        d = raw_class_dir / q
        return d if d.is_dir() else None
    for name in ("JPEG_IMAGES", "IMAGES"):
        d = raw_class_dir / name
        if d.is_dir():
            return d
    return raw_class_dir if raw_class_dir.is_dir() else None
