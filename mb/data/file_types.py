"""
Media suffix conventions for the data pipeline.

Configured lists come from ``data.image_types`` in :mod:`mb.pipeline_config` (see
default pipeline YAML). Use :func:`configured_media_suffixes` for scanning
sources and staging trees (gather, convert inputs, deduplicate, upscale, snapshot
helpers). Use :func:`normalized_jpeg_suffixes` where the pipeline only deals with
JPEG outputs (e.g. gather copy targets, convert JPEG detection).
"""

from __future__ import annotations

from typing import FrozenSet, Optional


def _normalize_suffix(entry: str) -> str:
    s = str(entry).strip().lower()
    if not s.startswith("."):
        s = "." + s
    return s


def configured_media_suffixes() -> FrozenSet[str]:
    """
    Lowercase suffixes (with leading dot) for pipeline media file scans.

    Backed by the active pipeline ``data.image_types`` list.
    """
    from mb.pipeline_config import get_pipeline_config

    raw = get_pipeline_config().get("data.image_types") or []
    return frozenset(_normalize_suffix(x) for x in raw if str(x).strip())


def configured_video_suffixes() -> FrozenSet[str]:
    """
    Lowercase suffixes for video files (``data.video_types`` in pipeline YAML).

    Used when scanning for image-classification sources that may include clips.
    """
    from mb.pipeline_config import get_pipeline_config

    raw = get_pipeline_config().get("data.video_types") or []
    return frozenset(_normalize_suffix(x) for x in raw if str(x).strip())


def configured_gather_scan_suffixes(model_type: Optional[str] = None) -> FrozenSet[str]:
    """
    Extensions to scan for gather / dedupe-style discovery.

    For ``image_classification``, includes :func:`configured_video_suffixes` in addition
    to :func:`configured_media_suffixes`.
    """
    from mb.pipeline_config import get_pipeline_config

    mt = model_type
    if mt is None:
        mt = get_pipeline_config().get("model.default_type", "image_classification")
    base = set(configured_media_suffixes())
    if mt == "image_classification":
        base |= set(configured_video_suffixes())
    return frozenset(base)


def normalized_jpeg_suffixes() -> FrozenSet[str]:
    """JPEG family suffixes for normalized outputs and gather copy targets."""
    return frozenset({".jpg", ".jpeg"})
