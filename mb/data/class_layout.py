"""
Class folders under ``raw_data`` / gather sources: discovery and qualifiers.

Pipeline keys (see :mod:`mb.pipeline_config`): ``data.class_names`` (optional list),
``data.class_qualifying_subdir`` (optional str). When the qualifier is set, only
directories containing that immediate child folder count as class (or gather) roots.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mb.utils.logging_setup import get_logger
logger = get_logger(__name__)

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

# Cache for :func:`discover_class_names`, keyed on ``(root, explicit, qualifier)``.
#
# The cache does not expire based on how much wall-clock time elapses between calls —
# two calls for the same root can legitimately be seconds (or more) apart (e.g. a
# main-thread precheck followed by the same check re-run on a worker thread once it
# starts, with slow unrelated work such as importing TensorFlow/PyTorch happening in
# between). A fixed debounce window just turns into a guessing game against whatever
# is slow that day. Instead, the scan result is treated as valid until something that
# could plausibly change the class-folder layout happens, at which point the caller is
# expected to invalidate it via :func:`clear_class_discovery_cache` (the UI does this
# after a data command finishes). ``DISCOVERY_CACHE_MAX_AGE_SECONDS`` is only a safety
# net bounding staleness if an invalidation is ever missed (e.g. the directory was
# changed by something outside the app while it stayed open).
DISCOVERY_CACHE_MAX_AGE_SECONDS = 300.0

_DiscoveryCacheKey = Tuple[str, Optional[Tuple[str, ...]], Optional[str]]
_discovery_cache: Dict[_DiscoveryCacheKey, Tuple[float, List[str]]] = {}


def clear_class_discovery_cache() -> None:
    """Drop all cached :func:`discover_class_names` results.

    Call this after any operation that can change which directories under a given
    root qualify as class folders (e.g. gather/convert creating new bucket dirs), and
    in tests that need a guaranteed fresh scan.
    """
    _discovery_cache.clear()


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

    Results are cached per ``(root, explicit, class_qualifying_subdir)`` until
    explicitly invalidated via :func:`clear_class_discovery_cache`, or until
    :data:`DISCOVERY_CACHE_MAX_AGE_SECONDS` elapses as a staleness safety net. This
    means repeated calls for the same root (e.g. from several UI widgets or code paths
    reacting to the same change) reuse the last scan instead of re-walking the
    filesystem and re-logging, regardless of how much unrelated wall-clock time passes
    between them.
    """
    root = Path(root)
    q = normalize_qualifying_subdir(class_qualifying_subdir)
    cache_key: _DiscoveryCacheKey = (
        str(root.resolve()) if root.exists() else str(root),
        tuple(str(n).strip() for n in explicit) if explicit else None,
        q,
    )
    now = time.monotonic()
    cached = _discovery_cache.get(cache_key)
    if cached is not None and (now - cached[0]) < DISCOVERY_CACHE_MAX_AGE_SECONDS:
        logger.debug(f"Using cached class names for root: {root}")
        return list(cached[1])

    logger.info(f"Discovering class names for root: {root}")

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
        _discovery_cache[cache_key] = (now, out)
        return out

    if not root.is_dir():
        _discovery_cache[cache_key] = (now, [])
        return []

    names: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if dir_qualifies_as_class(p, q):
            names.append(p.name)
    _discovery_cache[cache_key] = (now, names)
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
    logger.debug(f"Discovering raw data bucket names for: {raw_data_root}")
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
    logger.debug(f"Discovering review bucket names for: {review_root}")
    names = discover_class_names(
        review_root,
        explicit=explicit,
        class_qualifying_subdir=class_qualifying_subdir,
    )
    return [n for n in names if n not in _REVIEW_NON_CATEGORY_SUBDIRS]


def layout_dict_for_discovery() -> Dict[str, Any]:
    """``class_names`` + ``class_qualifying_subdir`` from the active pipeline config."""
    from mb.pipeline_config import data_class_layout_defaults

    logger.info("Creating layout dictionary for discovery")
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

    Prefer post-convert JPEG staging under each class folder first:
    ``CONVERTED`` then legacy ``JPEG_IMAGES``.

    If neither exists, and *class_qualifying_subdir* is set, use
    ``raw_class_dir / <qualifier>`` if it exists.

    Last fallback when qualifier is unset: ``IMAGES`` then *raw_class_dir*
    (images may live directly in the class folder).
    """
    logger.info(f"Resolving class media dir for: {raw_class_dir} (qualifier: {class_qualifying_subdir})")
    raw_class_dir = Path(raw_class_dir)
    for name in (CONVERTED_MEDIA_SUBDIR, LEGACY_CONVERTED_MEDIA_SUBDIR):
        d = raw_class_dir / name
        if d.is_dir():
            return d
    q = normalize_qualifying_subdir(class_qualifying_subdir)
    if q:
        d = raw_class_dir / q
        return d if d.is_dir() else None
    for name in ("IMAGES",):
        d = raw_class_dir / name
        if d.is_dir():
            return d
    return raw_class_dir if raw_class_dir.is_dir() else None
