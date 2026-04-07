"""
Heuristic disk-space checks before ``convert`` and ``create-dataset``.

Estimates how much *additional* space the next step is likely to need on the target
volume (rough JPEG/copy heuristics), compares to :func:`shutil.disk_usage`, and
records a fingerprint of source files so repeated runs can reuse estimates via
the unified snapshot and a small cache file under the raw data root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shutil

from mb.data.class_layout import (
    discover_class_names,
    normalize_qualifying_subdir,
    POST_CONVERT_SUBDIR_NAMES,
)
from mb.data.file_types import configured_media_suffixes, configured_video_suffixes
from mb.data.media_utils import classify_convert_source
from mb.models.types import ModelType, VisualMediaSourceType
from mb.pipeline_config import get_pipeline_config
from mb.utils.logging_setup import get_logger
from mb.utils.snapshot import UnifiedSnapshot, find_latest_unified_snapshot_path

logger = get_logger("mb.space_estimate")

# Heuristic overhead on top of summed source sizes (new JPEGs beside originals).
_CONVERT_SIZE_FACTOR = 1.12
# Per video / animated-GIF frame output + review copy (bytes), when source is not a simple still.
_EXTRACT_OUTPUT_BYTES = 4 * 1024 * 1024
# Create-dataset: copies one train file per image + test split moves (same total bytes ~ input sum).
_DATASET_COPY_FACTOR = 1.08

SPACE_CACHE_FILENAME = ".mb_space_estimate.json"


def format_bytes(n: int) -> str:
    """Human-readable size (binary units)."""
    if n < 0:
        n = 0
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


@dataclass
class SpaceEstimateReport:
    """Result of a space check for one operation."""

    operation: str  # "convert" | "create_dataset"
    fingerprint: str
    estimated_need_bytes: int
    free_bytes: int
    target_path: str
    ok: bool
    file_count: int
    source_total_bytes: int
    message: str
    computed_at: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["formatted_need"] = format_bytes(self.estimated_need_bytes)
        d["formatted_free"] = format_bytes(self.free_bytes)
        return d


def _disk_free_bytes(path: Path) -> int:
    try:
        logger.info(f"Checking disk free space for: {path}")
        return int(shutil.disk_usage(path).free)
    except OSError as e:
        logger.error(f"Error checking disk free space for: {path}: {e}")
        return 0


def _fingerprint_files(paths: List[Path], base: Path) -> str:
    """Stable hash from relative path, size, and mtime (ns)."""
    logger.info(f"Fingerprinting files: {paths} (base: {base})")
    h = hashlib.sha256()
    rows: List[Tuple[str, int, int]] = []
    for p in sorted(paths, key=lambda x: str(x).lower()):
        try:
            st = p.stat()
            rel = str(p.resolve().relative_to(base.resolve()))
            rows.append((rel.replace("\\", "/"), st.st_size, st.st_mtime_ns))
        except (OSError, ValueError):
            continue
    for rel, sz, mt in rows:
        h.update(f"{rel}|{sz}|{mt}\n".encode("utf-8", errors="replace"))
    h.update(str(len(rows)).encode())
    return h.hexdigest()


def _scan_suffixes_for_model(model_type: ModelType) -> List[str]:
    exts = set(configured_media_suffixes())
    if model_type == ModelType.IMAGE_CLASSIFICATION:
        exts |= set(configured_video_suffixes())
    return sorted(exts)


def _find_convert_paths_for_class(class_dir: Path, suffixes: List[str]) -> List[Path]:
    """Mirror :meth:`ImageConverter.find_image_files` without instantiating the full converter."""
    image_files: List[Path] = []
    post_convert_roots = [class_dir / n for n in POST_CONVERT_SUBDIR_NAMES]

    def _under_post_convert(p: Path) -> bool:
        for root in post_convert_roots:
            if root.exists() and p.is_relative_to(root):
                return True
        return False

    try:
        subdirs = [
            d
            for d in class_dir.iterdir()
            if d.is_dir() and d.name not in POST_CONVERT_SUBDIR_NAMES
        ]
        if subdirs:
            for subdir in subdirs:
                for ext in suffixes:
                    for file_path in subdir.rglob(f"*{ext}"):
                        image_files.append(file_path)
        else:
            for ext in suffixes:
                for file_path in class_dir.glob(f"*{ext}"):
                    if _under_post_convert(file_path):
                        continue
                    image_files.append(file_path)
    except OSError:
        pass
    return image_files


def collect_convert_source_paths(raw_data_dir: Path, model_type: ModelType) -> List[Path]:
    """All media paths that convert would process (per class discovery)."""
    logger.info(f"Collecting convert source paths for: {raw_data_dir}")
    raw_data_dir = Path(raw_data_dir)
    pc = get_pipeline_config()
    qual = normalize_qualifying_subdir(pc.get("data.class_qualifying_subdir"))
    ex = pc.get("data.class_names")
    explicit_list = ex if isinstance(ex, list) else None
    class_names = discover_class_names(
        raw_data_dir,
        explicit=explicit_list,
        class_qualifying_subdir=qual,
    )
    suffixes = _scan_suffixes_for_model(model_type)
    out: List[Path] = []
    for name in class_names:
        class_dir = raw_data_dir / name
        if not class_dir.is_dir():
            continue
        out.extend(_find_convert_paths_for_class(class_dir, suffixes))
    return out


def estimate_convert_additional_bytes(paths: List[Path], model_type: ModelType) -> Tuple[int, int, int]:
    """
    Rough additional bytes needed on the raw-data volume (new CONVERTED JPEGs + review copies).

    Returns:
        (estimated_bytes, file_count, sum of raw source file sizes for stats)
    """
    logger.info(f"Estimating convert additional bytes for: {paths}")
    total_source = 0
    additional = 0
    n = 0
    for p in paths:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        n += 1
        total_source += sz
        st = classify_convert_source(p, model_type=model_type)
        if st == VisualMediaSourceType.STATIC:
            # New JPEG alongside original; often similar or smaller than source.
            additional += int(min(sz, 80 * 1024 * 1024) * 0.92)
        elif st in (VisualMediaSourceType.VIDEO_EXTRACT, VisualMediaSourceType.ANIMATED_GIF_EXTRACT):
            additional += _EXTRACT_OUTPUT_BYTES
        else:
            additional += int(sz * 0.5)
    # Small metadata / filesystem overhead
    additional = int(additional * 1.02)
    return max(additional, 64 * 1024), n, total_source


def estimate_convert(
    raw_data_dir: Path,
    model_type: ModelType,
) -> SpaceEstimateReport:
    logger.info(f"Estimating convert space for: {raw_data_dir}")
    paths = collect_convert_source_paths(raw_data_dir, model_type)
    need, n, src_sum = estimate_convert_additional_bytes(paths, model_type)
    fp = _fingerprint_files(paths, Path(raw_data_dir))
    free = _disk_free_bytes(Path(raw_data_dir))
    ok = free >= need
    msg = (
        f"Convert: ~{format_bytes(need)} additional space likely on this drive "
        f"({n} source files, ~{format_bytes(src_sum)} total source bytes). "
        f"Free: {format_bytes(free)}."
    )
    if not ok:
        msg += f" Short by ~{format_bytes(need - free)}."
    return SpaceEstimateReport(
        operation="convert",
        fingerprint=fp,
        estimated_need_bytes=need,
        free_bytes=free,
        target_path=str(Path(raw_data_dir).resolve()),
        ok=ok,
        file_count=n,
        source_total_bytes=src_sum,
        message=msg,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def collect_dataset_jpeg_paths(raw_data_dir: Path) -> List[Path]:
    """JPEGs under each class's resolved media dir (same as DatasetCreator)."""
    raw_data_dir = Path(raw_data_dir)
    pc = get_pipeline_config()
    qual = normalize_qualifying_subdir(pc.get("data.class_qualifying_subdir"))
    ex = pc.get("data.class_names")
    explicit_list = ex if isinstance(ex, list) else None
    class_names = discover_class_names(
        raw_data_dir,
        explicit=explicit_list,
        class_qualifying_subdir=qual,
    )
    from mb.data.class_layout import resolve_class_media_dir

    out: List[Path] = []
    for name in class_names:
        raw_class_dir = raw_data_dir / name
        media_dir = resolve_class_media_dir(raw_class_dir, qual)
        if media_dir is None or not media_dir.exists():
            continue
        out.extend(media_dir.glob("*.jpg"))
        out.extend(media_dir.glob("*.jpeg"))
    return out


def estimate_create_dataset(
    raw_data_dir: Path,
    data_dir: Path,
) -> SpaceEstimateReport:
    paths = collect_dataset_jpeg_paths(raw_data_dir)
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    need = int(total * _DATASET_COPY_FACTOR)
    fp = _fingerprint_files(paths, Path(raw_data_dir))
    free = _disk_free_bytes(Path(data_dir))
    ok = free >= need
    n = len(paths)
    msg = (
        f"Create-dataset: ~{format_bytes(need)} likely on output drive "
        f"({n} JPEGs from raw, ~{format_bytes(total)} bytes). "
        f"Free: {format_bytes(free)}."
    )
    if not ok:
        msg += f" Short by ~{format_bytes(need - free)}."
    return SpaceEstimateReport(
        operation="create_dataset",
        fingerprint=fp,
        estimated_need_bytes=need,
        free_bytes=free,
        target_path=str(Path(data_dir).resolve()),
        ok=ok,
        file_count=n,
        source_total_bytes=total,
        message=msg,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _cache_path(raw_data_dir: Path) -> Path:
    return Path(raw_data_dir) / SPACE_CACHE_FILENAME


def load_space_cache(raw_data_dir: Path) -> Optional[Dict[str, Any]]:
    p = _cache_path(raw_data_dir)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            logger.info(f"Loading space cache from: {p}")
            return json.load(f)
    except Exception:
        return None


def save_space_cache(raw_data_dir: Path, cache: Dict[str, Any]) -> None:
    p = _cache_path(raw_data_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        logger.warning("Could not write space estimate cache %s: %s", p, e)


def snapshot_has_valid_convert_estimate(
    snapshot: Optional[UnifiedSnapshot], fingerprint: str
) -> bool:
    logger.info(f"Checking if snapshot has valid convert estimate for fingerprint")
    if not snapshot or not getattr(snapshot, "space_estimates", None):
        return False
    se = snapshot.space_estimates
    c = se.get("convert") if isinstance(se, dict) else None
    if not isinstance(c, dict):
        return False
    return c.get("fingerprint") == fingerprint


def snapshot_has_valid_dataset_estimate(
    snapshot: Optional[UnifiedSnapshot],
    fingerprint: str,
    data_dir: Path,
) -> bool:
    logger.info(f"Checking if snapshot has valid dataset estimate for fingerprint (data_dir: {data_dir})")
    if not snapshot or not getattr(snapshot, "space_estimates", None):
        return False
    se = snapshot.space_estimates
    d = se.get("create_dataset") if isinstance(se, dict) else None
    if not isinstance(d, dict):
        return False
    if d.get("fingerprint") != fingerprint:
        return False
    try:
        return Path(str(d.get("data_dir", ""))).resolve() == Path(data_dir).resolve()
    except OSError:
        return False


def run_convert_estimate(
    raw_data_dir: Path,
    model_type: ModelType,
    *,
    snapshot: Optional[UnifiedSnapshot] = None,
) -> SpaceEstimateReport:
    """Compute convert estimate; reuse snapshot/cache fingerprint match when possible."""
    logger.info("Scanning raw sources for convert space estimate…")
    raw_data_dir = Path(raw_data_dir)
    if snapshot is None:
        snap_path = find_latest_unified_snapshot_path([raw_data_dir])
        if snap_path:
            snapshot = UnifiedSnapshot.load(snap_path)
    paths = collect_convert_source_paths(raw_data_dir, model_type)
    fp = _fingerprint_files(paths, Path(raw_data_dir))
    cache = load_space_cache(raw_data_dir)
    if cache and cache.get("convert", {}).get("fingerprint") == fp:
        need = int(cache["convert"]["estimated_need_bytes"])
        n = int(cache["convert"].get("file_count", len(paths)))
        src_sum = int(cache["convert"].get("source_total_bytes", 0))
    elif snapshot_has_valid_convert_estimate(snapshot, fp):
        logger.info(f"Using cached convert estimate for fingerprint: {fp}")
        c = snapshot.space_estimates["convert"]
        need = int(c["estimated_need_bytes"])
        n = int(c.get("file_count", len(paths)))
        src_sum = int(c.get("source_total_bytes", 0))
    else:
        need, n, src_sum = estimate_convert_additional_bytes(paths, model_type)
    free = _disk_free_bytes(Path(raw_data_dir))
    ok = free >= need
    msg = (
        f"Convert: ~{format_bytes(need)} additional space likely on this drive "
        f"({n} source files, ~{format_bytes(src_sum)} total source bytes). "
        f"Free: {format_bytes(free)}."
    )
    if not ok:
        msg += f" Short by ~{format_bytes(need - free)}."
    logger.info(f"Convert space estimate: {msg}")
    rep = SpaceEstimateReport(
        operation="convert",
        fingerprint=fp,
        estimated_need_bytes=need,
        free_bytes=free,
        target_path=str(Path(raw_data_dir).resolve()),
        ok=ok,
        file_count=n,
        source_total_bytes=src_sum,
        message=msg,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    # Update cache
    full_cache = cache if isinstance(cache, dict) else {}
    full_cache["convert"] = {
        "fingerprint": fp,
        "estimated_need_bytes": need,
        "file_count": n,
        "source_total_bytes": src_sum,
        "computed_at": rep.computed_at,
    }
    save_space_cache(raw_data_dir, full_cache)
    return rep


def merge_convert_estimate_into_snapshot(snapshot: UnifiedSnapshot, report: SpaceEstimateReport) -> None:
    if not hasattr(snapshot, "space_estimates") or snapshot.space_estimates is None:
        snapshot.space_estimates = {}
    snapshot.space_estimates["convert"] = {
        "fingerprint": report.fingerprint,
        "estimated_need_bytes": report.estimated_need_bytes,
        "file_count": report.file_count,
        "source_total_bytes": report.source_total_bytes,
        "computed_at": report.computed_at,
    }


def run_create_dataset_estimate(
    raw_data_dir: Path,
    data_dir: Path,
    *,
    snapshot: Optional[UnifiedSnapshot] = None,
) -> SpaceEstimateReport:
    logger.info("Scanning JPEGs for create-dataset space estimate…")
    raw_data_dir = Path(raw_data_dir)
    paths = collect_dataset_jpeg_paths(raw_data_dir)
    fp = _fingerprint_files(paths, Path(raw_data_dir))
    cache = load_space_cache(raw_data_dir)
    ds_cache = (cache or {}).get("create_dataset") if isinstance(cache, dict) else None
    if (
        isinstance(ds_cache, dict)
        and ds_cache.get("fingerprint") == fp
        and str(ds_cache.get("data_dir", "")).lower() == str(data_dir).lower()
    ):
        need = int(ds_cache["estimated_need_bytes"])
        total = int(ds_cache.get("source_total_bytes", 0))
        n = int(ds_cache.get("file_count", len(paths)))
    elif snapshot_has_valid_dataset_estimate(snapshot, fp, Path(data_dir)):
        d = snapshot.space_estimates["create_dataset"]
        need = int(d["estimated_need_bytes"])
        total = int(d.get("source_total_bytes", 0))
        n = int(d.get("file_count", len(paths)))
    else:
        total = 0
        for p in paths:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        need = int(total * _DATASET_COPY_FACTOR)
        n = len(paths)
    free = _disk_free_bytes(Path(data_dir))
    ok = free >= need
    msg = (
        f"Create-dataset: ~{format_bytes(need)} likely on output drive "
        f"({n} JPEGs from raw, ~{format_bytes(total)} bytes). "
        f"Free: {format_bytes(free)}."
    )
    if not ok:
        msg += f" Short by ~{format_bytes(need - free)}."
    rep = SpaceEstimateReport(
        operation="create_dataset",
        fingerprint=fp,
        estimated_need_bytes=need,
        free_bytes=free,
        target_path=str(Path(data_dir).resolve()),
        ok=ok,
        file_count=n,
        source_total_bytes=total,
        message=msg,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    full_cache = cache if isinstance(cache, dict) else {}
    full_cache["create_dataset"] = {
        "fingerprint": fp,
        "data_dir": str(data_dir),
        "estimated_need_bytes": need,
        "file_count": n,
        "source_total_bytes": total,
        "computed_at": rep.computed_at,
    }
    save_space_cache(raw_data_dir, full_cache)
    if snapshot is not None:
        if not hasattr(snapshot, "space_estimates") or snapshot.space_estimates is None:
            snapshot.space_estimates = {}
        snapshot.space_estimates["create_dataset"] = {
            "fingerprint": fp,
            "data_dir": str(data_dir),
            "estimated_need_bytes": need,
            "file_count": n,
            "source_total_bytes": total,
            "computed_at": rep.computed_at,
        }
    return rep


def check_convert_allowed(
    raw_data_dir: Path,
    model_type: ModelType,
    *,
    snapshot: Optional[UnifiedSnapshot] = None,
    skip_space_check: bool = False,
) -> Tuple[bool, SpaceEstimateReport]:
    """
    Run convert space estimate; return (allowed, report).

    If *skip_space_check* is True, always returns allowed=True (still computes report for logging).
    """
    report = run_convert_estimate(raw_data_dir, model_type, snapshot=snapshot)
    logger.info(report.message)
    if skip_space_check:
        return True, report
    return report.ok, report


def check_create_dataset_allowed(
    raw_data_dir: Path,
    data_dir: Path,
    *,
    snapshot: Optional[UnifiedSnapshot] = None,
    skip_space_check: bool = False,
) -> Tuple[bool, SpaceEstimateReport]:
    report = run_create_dataset_estimate(raw_data_dir, data_dir, snapshot=snapshot)
    logger.info(report.message)
    if skip_space_check:
        return True, report
    return report.ok, report
