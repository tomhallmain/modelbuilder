"""Small shared helpers used by ``mb.utils`` (e.g. translations) and path utilities."""

from __future__ import annotations

import hashlib
import locale
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Windows: one filename component normally ≤ 255; legacy full path ≤ 260 without ``\\?\`` prefix.
_WIN_MAX_COMPONENT = 255
_WIN_MAX_PATH = 260


def _normalize_output_suffix(suffix: str) -> str:
    """Ensure *suffix* is a non-empty extension with a leading dot (e.g. ``.jpg``)."""
    s = suffix.strip()
    if not s:
        raise ValueError("output_suffix must be non-empty")
    return s if s.startswith(".") else f".{s}"


def _affix_suffix_char_count(output_suffix: str) -> int:
    """Characters after the stem: ``_`` + 8 hex + *output_suffix*."""
    return 1 + 8 + len(_normalize_output_suffix(output_suffix))


def _resolved_path_str(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _shorten_stem_for_windows_paths(
    stem: str,
    affix: str,
    output_dirs: Tuple[Path, ...],
    output_suffix: str,
) -> str:
    """
    Trim *stem* so each ``dir / {stem}_{affix}{output_suffix}`` fits under NTFS component and MAX_PATH rules.
    """
    ext = _normalize_output_suffix(output_suffix)
    suffix = f"_{affix}{ext}"
    max_stem_by_component = _WIN_MAX_COMPONENT - len(suffix)
    stem_cur = stem[:max_stem_by_component] if len(stem) > max_stem_by_component else stem
    max_path_chars = _WIN_MAX_PATH - 1  # usable length without ``\\?\``

    def fits(st: str) -> bool:
        base = f"{st}{suffix}"
        if len(base) > _WIN_MAX_COMPONENT:
            return False
        for d in output_dirs:
            full = _resolved_path_str(Path(d) / base)
            if len(full) > max_path_chars:
                return False
        return True

    while stem_cur and not fits(stem_cur):
        stem_cur = stem_cur[:-1]
    if not fits(stem_cur):
        logger.warning(
            "Output path(s) exceed Windows MAX_PATH even with minimal stem; using shortest basename"
        )
        stem_cur = ""
    return stem_cur


def convert_output_filename(
    source_path: Path,
    *,
    output_suffix: str = ".jpg",
    output_dir: Optional[Path] = None,
    also_under_dirs: Optional[Tuple[Path, ...]] = None,
) -> str:
    """
    Basename for converted media: ``{stem}_{last_8_hex_chars_of_sha256}{output_suffix}``.

    The hash is of the UTF-8 encoding of :meth:`pathlib.Path.resolve` in POSIX form (stable
    across runs for the same file path).

    *output_suffix* is the target extension, with or without a leading dot (e.g. ``.jpg`` or ``png``).

    The stem is truncated as needed so the final name stays within a single component's
    limit (255 on Windows NTFS). On Windows, if *output_dir* and optionally *also_under_dirs*
    are given, the stem is shortened further so each full path stays below the legacy 260
    character limit (unless long paths are enabled in the environment).
    """
    ext = _normalize_output_suffix(output_suffix)
    p = Path(source_path)
    try:
        normalized = p.resolve().as_posix()
    except OSError:
        normalized = str(p)
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
    affix = digest[-8:]

    max_stem = _WIN_MAX_COMPONENT - _affix_suffix_char_count(ext)
    stem = p.stem[:max_stem] if len(p.stem) > max_stem else p.stem

    if sys.platform == "win32":
        dirs: List[Path] = []
        if output_dir is not None:
            dirs.append(Path(output_dir))
        if also_under_dirs:
            dirs.extend(Path(x) for x in also_under_dirs)
        if dirs:
            stem = _shorten_stem_for_windows_paths(stem, affix, tuple(dirs), ext)

    return f"{stem}_{affix}{ext}"


def convert_output_jpeg_filename(
    source_path: Path,
    *,
    output_dir: Optional[Path] = None,
    also_under_dirs: Optional[Tuple[Path, ...]] = None,
) -> str:
    """Basename for JPEG outputs under ``CONVERTED/``; same as :func:`convert_output_filename` with ``.jpg``."""
    return convert_output_filename(
        source_path,
        output_suffix=".jpg",
        output_dir=output_dir,
        also_under_dirs=also_under_dirs,
    )


_JPEG_SOURCE_SUFFIXES = frozenset({".jpg", ".jpeg", ".jpe"})


def plain_still_jpeg_basename(source_stem: str) -> str:
    """
    Basename ``{stem}.jpg`` for a still-image convert output, trimmed to fit one NTFS component.

    Used together with :func:`assign_still_convert_output_basenames` so one source per stem group
    can use this plain name; additional sources with the same stem use :func:`convert_output_jpeg_filename`.
    """
    if not source_stem:
        raise ValueError("plain_still_jpeg_basename requires a non-empty stem")
    max_plain_stem = _WIN_MAX_COMPONENT - len(".jpg")
    stem = source_stem[:max_plain_stem] if len(source_stem) > max_plain_stem else source_stem
    return f"{stem}.jpg"


def assign_still_convert_output_basenames(
    source_paths: Sequence[Path],
    *,
    output_dir: Optional[Path] = None,
    also_under_dirs: Optional[Tuple[Path, ...]] = None,
) -> Dict[Path, str]:
    """
    Map each still-image source path to its ``CONVERTED/`` JPEG basename.

    For each distinct :attr:`~pathlib.PurePath.stem`, **one** file (preferring an existing JPEG
    source over other extensions, then lexicographic path) is assigned ``{stem}.jpg`` (see
    :func:`plain_still_jpeg_basename`). Every other source with that stem is assigned
    :func:`convert_output_jpeg_filename` (path-hash affix) so names stay unique.

    Stems longer than a single component allows, or an empty stem, fall back to hash-only names for
    every source in that group.
    """
    paths = list(source_paths)
    if not paths:
        return {}
    max_plain_stem = _WIN_MAX_COMPONENT - len(".jpg")
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_stem[p.stem].append(p)
    out: Dict[Path, str] = {}
    for stem, group in by_stem.items():
        if not stem or len(stem) > max_plain_stem:
            for p in group:
                out[p] = convert_output_jpeg_filename(
                    p, output_dir=output_dir, also_under_dirs=also_under_dirs
                )
            continue
        ranked = sorted(
            group,
            key=lambda p: (
                0 if p.suffix.lower() in _JPEG_SOURCE_SUFFIXES else 1,
                str(p),
            ),
        )
        plain_basename = plain_still_jpeg_basename(stem)
        out[ranked[0]] = plain_basename
        for p in ranked[1:]:
            out[p] = convert_output_jpeg_filename(
                p, output_dir=output_dir, also_under_dirs=also_under_dirs
            )
    return out


class Utils:
    @staticmethod
    def get_default_user_language() -> str:
        try:
            loc = locale.getdefaultlocale()[0]
            if not loc:
                return "en"
            if "_" in loc:
                return loc.split("_")[0]
            return loc
        except Exception:
            return "en"
