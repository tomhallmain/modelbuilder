"""
Find ``.jpg`` / ``.jpeg`` files whose bytes are not JPEG, and repair class-folder layouts.

**CLI:** ``mb data fix-jpeg-extension-mismatch``; ``python -m mb.data.find_jpeg_extension_mismatches``
delegates via :func:`mb.cli.run_data_subcommand_cli`.

Mislabeled sources are renamed in place to match the detected container (e.g. ``.gif``). Outputs that
were produced by :mod:`mb.data.convert` via the JPEG copy path (non-JPEG bytes under a ``.jpg`` name
in ``CONVERTED`` / ``small_images_review``) are removed only after a replacement JPEG is written
successfully.

When a unified snapshot (``snapshot_<run_id>.json``) exists under the raw data root, each repaired
file updates **only** that image’s ``original`` and ``converted`` fields to match the renamed source
and the new ``CONVERTED`` JPEG—``dataset`` / ``training`` are left unchanged. Step timing and any
errors are recorded under ``fix-jpeg-extension-mismatch`` in the snapshot’s ``step_errors`` map.

With ``dry_run=True``, optional ``dry_run_pillow`` / ``dry_run_quiet`` mirror list-mode switches:
optional Pillow metadata (with JSON lines) and suppression of verbose per-file log lines.

With ``json_lines=True`` (CLI ``--json``), emit newline-delimited JSON for each relevant mismatch to
stdout (or the step logger when ``json_lines_to_logger`` is true). Without ``verbose``, only
**actionable** mismatches are emitted (policy-skipped PNG/WebP/BMP/TIFF are omitted); with
``verbose``, every mismatch is emitted. The same rule applies in live repair: JSON lines for skips
only when ``verbose`` is true; successful repairs always emit one JSON line when ``json_lines`` is
true.

Dry-run and live repair both use **one** :meth:`~mb.data.convert.ImageConverter.find_image_files`
scan per class, one ``assign_still_convert_output_basenames`` map, and the same sorted pass over
mismatches (live adds writes, renames, and snapshot updates only).

By default, mislabeled ``.jpg`` whose bytes are PNG/WebP/BMP/TIFF are only counted and summarized per
class; pass ``include_static_format_mismatches=True`` (CLI: ``--include-static-format-mismatches``)
to rename and rebuild them like other non-JPEG containers. GIF and image-classification multi-frame
GIF handling are unchanged.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from mb.cancellation import check_cancel_event
from mb.data.class_layout import CONVERTED_MEDIA_SUBDIR, VISUAL_MEDIA_REVIEW_SUBDIR
from mb.data.class_layout import discover_class_names, layout_dict_for_discovery
from mb.data.convert import ImageConverter
from mb.data.media_utils import (
    extract_random_gif_frame_to_jpeg,
    pil_gif_frame_count,
    pil_image_to_jpeg_normalized,
)
from mb.models.types import ModelBuildStepCommand, ModelType
from mb.pipeline_config import get_pipeline_config
from mb.utils.logging_setup import log_completion_info, log_startup_info, setup_logging
from mb.utils.utils import assign_still_convert_output_basenames, convert_output_jpeg_filename

from mb.utils.snapshot import (
    calculate_file_hash,
    find_unified_snapshot,
    preload_gather_cache,
    save_unified_snapshot,
    set_step_errors_for_invocation,
)

logger = setup_logging(script_name="fix_jpeg_mismatch")

JPEG_MAGIC = b"\xff\xd8\xff"

# Mislabeled ``.jpg`` / ``.jpeg`` whose bytes are PNG/WebP/BMP/TIFF: harmless for most PIL-based training
# (convert copies or re-encodes). Skipped unless *include_static_format_mismatches* is true.
_STATIC_FORMAT_EXTENSIONS_DEFAULT_SKIP: frozenset[str] = frozenset({".png", ".webp", ".bmp", ".tiff"})


def _static_format_skipped_by_policy(new_ext: str, include_static_format_mismatches: bool) -> bool:
    if include_static_format_mismatches:
        return False
    e = (new_ext or "").lower()
    return e in _STATIC_FORMAT_EXTENSIONS_DEFAULT_SKIP


def read_file_header(path: Path, n: int = 32) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def sniff_container(head: bytes) -> str:
    """Return a short label for the payload implied by the first bytes."""
    if len(head) < 3:
        return "empty_or_truncated"
    if head[:3] == JPEG_MAGIC:
        return "jpeg"
    if len(head) >= 6 and head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(head) >= 8 and head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if len(head) >= 2 and head[:2] == b"BM":
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "unknown"


def is_jpeg_magic(path: Path) -> bool:
    try:
        return sniff_container(read_file_header(path)) == "jpeg"
    except OSError:
        return False


def pillow_format_and_gif_info(path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return None, {"error": "Pillow not installed"}
    try:
        with Image.open(path) as im:
            fmt = im.format
            extra: Dict[str, Any] = {}
            if fmt == "GIF":
                info = getattr(im, "info", {}) or {}
                for key in ("version", "background", "loop", "transparency", "duration", "extension"):
                    if key in info:
                        val = info[key]
                        if isinstance(val, bytes):
                            extra[key] = val.decode("latin-1", errors="replace")
                        else:
                            extra[key] = val
            return fmt, extra
    except (OSError, UnidentifiedImageError) as e:
        return None, {"error": str(e)}


def extension_for_container(sniff: str, pil_format: Optional[str]) -> Optional[str]:
    mapping = {
        "gif": ".gif",
        "png": ".png",
        "webp": ".webp",
        "bmp": ".bmp",
        "tiff": ".tiff",
    }
    if sniff in mapping:
        return mapping[sniff]
    if pil_format:
        p = pil_format.upper()
        if p in ("JPEG", "MPO"):
            return ".jpg"
        if p == "PNG":
            return ".png"
        if p == "GIF":
            return ".gif"
        if p == "WEBP":
            return ".webp"
        if p == "BMP":
            return ".bmp"
        if p == "TIFF":
            return ".tiff"
    return None


def _mismatch_row_base(path: Path) -> Dict[str, Any]:
    head = read_file_header(path)
    kind = sniff_container(head)
    return {
        "path": str(path),
        "sniffed": kind,
        "header_hex": head[:16].hex(),
    }


def _merge_pillow_report(path: Path, row: Dict[str, Any]) -> None:
    fmt, extra = pillow_format_and_gif_info(path)
    row["pillow"] = {"format": fmt, **extra}


@dataclass
class RepairStats:
    classes_scanned: int = 0
    files_repaired: int = 0
    files_failed: int = 0
    dry_run_actions: int = 0
    mismatches_found: int = 0
    """All ``.jpg``/``.jpeg`` files whose bytes are not JPEG (including policy-skipped static formats)."""
    actionable_mismatches_found: int = 0
    """Subset of mismatches that would be repaired under the current policy (excludes skipped PNG/WebP/BMP/TIFF)."""
    skipped_static_format_by_class: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    snapshot_records_updated: int = 0
    snapshot_apply_errors: List[str] = field(default_factory=list)


def _posix_rel_str(path: Optional[str]) -> str:
    if not path:
        return ""
    return str(path).replace("\\", "/")


def _rel_under_raw(raw_data_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(raw_data_dir.resolve()).as_posix()


def _find_snapshot_record_by_original_path(
    snapshot: Any,
    rel_original_old: str,
) -> Optional[str]:
    """
    Return the dict key (original content hash) for the record whose ``original.path`` matches
    *rel_original_old* (POSIX, relative to raw data root).
    """
    want = _posix_rel_str(rel_original_old)
    for hash_key, rec in snapshot.images.items():
        orig = rec.get("original")
        if not isinstance(orig, dict):
            continue
        if _posix_rel_str(orig.get("path")) == want:
            return str(hash_key)
    return None


def _update_snapshot_record_after_repair(
    snapshot: Any,
    raw_data_dir: Path,
    class_name: str,
    rel_original_old: str,
    renamed_source: Path,
    converted_jpeg: Path,
    *,
    logger,
) -> None:
    """
    Update *only* ``original`` and ``converted`` for the image that was mislabeled as JPEG.

    Does not modify ``dataset`` or ``training``. Skips quietly if no matching record exists.
    """
    hash_key = _find_snapshot_record_by_original_path(snapshot, rel_original_old)
    if not hash_key:
        logger.debug(
            "Snapshot: no record with original.path=%r (skipping snapshot row update)",
            rel_original_old,
        )
        return
    rec = snapshot.images.get(hash_key)
    if not isinstance(rec, dict):
        return

    rel_new = _rel_under_raw(raw_data_dir, renamed_source)
    md5_c = calculate_file_hash(
        converted_jpeg,
        algorithm="md5",
        raw_data_dir=raw_data_dir,
        unified_snapshot=None,
        logger=logger,
    )
    sha_c = calculate_file_hash(
        converted_jpeg,
        algorithm="sha256",
        raw_data_dir=raw_data_dir,
        unified_snapshot=None,
        logger=logger,
    )
    if md5_c is None or sha_c is None:
        raise RuntimeError(f"could not hash converted output {converted_jpeg}")

    prior_orig = rec.get("original") if isinstance(rec.get("original"), dict) else {}
    prior_hash = prior_orig.get("hash")
    was_conv = bool((prior_hash or hash_key) != md5_c)

    rec["original"] = {
        "basename": renamed_source.name,
        "hash": prior_hash or hash_key,
        "path": rel_new,
        "format": renamed_source.suffix.lower(),
    }
    conv_rel = _rel_under_raw(raw_data_dir, converted_jpeg)
    rec["converted"] = {
        "class": class_name,
        "path": conv_rel,
        "basename": converted_jpeg.name,
        "md5": md5_c,
        "sha256": sha_c,
        "was_converted": was_conv,
    }
    snapshot.last_updated = datetime.now().isoformat()


def _gif_frame_count(path: Path) -> int:
    n = pil_gif_frame_count(path)
    if n is None:
        return 1
    return max(1, n)


def _unique_dest(path: Path) -> Path:
    if not path.exists():
        return path
    c = 1
    orig = path
    while path.exists():
        path = orig.with_name(f"{orig.stem}_{c}{orig.suffix}")
        c += 1
    return path


def _copy_visual_review_like_convert(target_jpeg: Path, review_dir: Path) -> Optional[Path]:
    """Mirror :meth:`mb.data.convert.ImageConverter.process_visual_extractions` review copy."""
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / target_jpeg.name
    r_counter = 1
    orig_review_stem = review_path.stem
    while review_path.exists():
        review_path = review_dir / f"{orig_review_stem}_{r_counter}.jpg"
        r_counter += 1
    shutil.copy2(target_jpeg, review_path)
    return review_path


def _emit_json_line(
    obj: Dict[str, Any],
    *,
    json_lines_to_logger: bool,
) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    if json_lines_to_logger:
        logger.info("%s", line)
    else:
        print(line, flush=True)


def repair_mislabeled_jpeg_extensions(
    raw_data_dir: Path,
    *,
    model_type: ModelType,
    dry_run: bool = False,
    json_lines: bool = False,
    dry_run_pillow: bool = False,
    dry_run_quiet: bool = False,
    json_lines_to_logger: bool = False,
    include_static_format_mismatches: bool = False,
    verbose: bool = False,
    cancel_event: Optional[threading.Event] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[bool, RepairStats]:
    """
    Walk pipeline class folders like convert: rename mislabeled ``.jpg`` sources, write real JPEGs,
    and remove stale copies under ``CONVERTED`` / ``small_images_review`` only after success.

    When *dry_run* is true, optional reporting flags apply:

    * *json_lines* — emit JSON objects (see module docstring). Uses stdout unless *json_lines_to_logger*
      is true (GUI / log capture). May be combined with live repair (not dry run).
    * *dry_run_pillow* — add a ``pillow`` key with format / GIF metadata (meaningful with *json_lines*;
      dry run only).
    * *dry_run_quiet* — skip verbose per-file :meth:`logging.Logger.info` lines when not using JSON.

    *verbose* — include policy-skipped static-format mismatches in JSON output and per-file text lines
    for those skips; log class progress for every folder. Suppressed when *dry_run_quiet* is true.

    *include_static_format_mismatches* — when false (default), mislabeled ``.jpg`` / ``.jpeg`` files whose
    bytes are PNG, WebP, BMP, or TIFF are counted and summarized per class but not renamed or repaired
    (GIF and animated-IC random-frame cases are still repaired). Set true to repair those as well.
    """
    if dry_run_pillow and not dry_run:
        raise ValueError("dry_run_pillow requires dry_run=True")
    if dry_run_quiet and not dry_run:
        raise ValueError("dry_run_quiet requires dry_run=True")
    if dry_run_pillow and not json_lines:
        raise ValueError("dry_run_pillow requires json_lines=True")

    log_startup_info(logger, "Repair mislabeled JPEG extensions (dry_run=%s)" % (dry_run,))
    stats = RepairStats()
    raw_data_dir = Path(raw_data_dir)
    rng = rng or random.Random()
    layout = layout_dict_for_discovery()
    pc = get_pipeline_config()
    ex = pc.get("data.class_names")
    explicit_list = ex if isinstance(ex, list) else None
    qual = layout.get("class_qualifying_subdir")
    class_names = discover_class_names(
        raw_data_dir,
        explicit=explicit_list,
        class_qualifying_subdir=qual,
    )
    if not class_names:
        logger.warning("No class directories found under %s", raw_data_dir)
        log_completion_info(logger, True, "no class folders to scan")
        return True, stats

    repair_started_iso: Optional[str] = None
    wall_t0: Optional[float] = None
    snapshot: Any = None
    if not dry_run:
        wall_t0 = time.perf_counter()
        repair_started_iso = datetime.now(timezone.utc).isoformat()
        snapshot = find_unified_snapshot([raw_data_dir], run_id=None, logger=logger)
        if snapshot is None:
            logger.warning(
                "No unified snapshot under %s; file repair will run but snapshot rows will not be updated. "
                "Run mb data convert first to create a snapshot if you need metadata alignment.",
                raw_data_dir,
            )
        else:
            preload_gather_cache(raw_data_dir)

    converter = ImageConverter(raw_data_dir=raw_data_dir, model_type=model_type)
    small_review_name = "small_images_review"

    for ci, class_name in enumerate(class_names):
        check_cancel_event(cancel_event)
        if verbose or ci % 2 == 0:
            logger.info("Class %s (%d/%d)", class_name, ci + 1, len(class_names))
        class_dir = raw_data_dir / class_name
        if not class_dir.is_dir():
            continue
        stats.classes_scanned += 1

        converted_dir = class_dir / CONVERTED_MEDIA_SUBDIR
        review_dir = class_dir / VISUAL_MEDIA_REVIEW_SUBDIR
        small_review_dir = class_dir / small_review_name

        if dry_run:
            image_files = converter.find_image_files(class_dir, log_scan=False)
            static_paths, _ = converter._split_static_and_extract(image_files)
            mismatches = [
                p
                for p in static_paths
                if p.suffix.lower() in (".jpg", ".jpeg") and not is_jpeg_magic(p)
            ]
            static_paths.sort(key=lambda x: str(x))
            basename_map = assign_still_convert_output_basenames(
                static_paths,
                output_dir=converted_dir,
            )
            for src in sorted(mismatches, key=lambda x: str(x)):
                stats.mismatches_found += 1
                expected_jpg_name = basename_map.get(src, "?")
                head = read_file_header(src)
                sniff = sniff_container(head)
                pil_fmt, _ = pillow_format_and_gif_info(src)
                new_ext = extension_for_container(sniff, pil_fmt) or "?"
                n_gif = (
                    _gif_frame_count(src)
                    if (pil_fmt == "GIF" or sniff == "gif")
                    else 1
                )
                animated_ic = (
                    model_type == ModelType.IMAGE_CLASSIFICATION
                    and (pil_fmt == "GIF" or sniff == "gif")
                    and n_gif > 1
                )
                skipped_by_policy = (
                    new_ext != "?"
                    and _static_format_skipped_by_policy(new_ext, include_static_format_mismatches)
                    and not animated_ic
                )
                if skipped_by_policy:
                    stats.skipped_static_format_by_class[class_name] = (
                        stats.skipped_static_format_by_class.get(class_name, 0) + 1
                    )
                else:
                    stats.actionable_mismatches_found += 1
                    stats.dry_run_actions += 1
                if json_lines and (verbose or not skipped_by_policy):
                    row: Dict[str, Any] = {
                        **_mismatch_row_base(src),
                        "class": class_name,
                        "expected_converted_basename": expected_jpg_name,
                        "inferred_extension": new_ext,
                        "animated_ic_random_frame": animated_ic,
                        "skipped_by_policy": skipped_by_policy,
                        "dry_run": True,
                    }
                    if dry_run_pillow:
                        _merge_pillow_report(src, row)
                    _emit_json_line(row, json_lines_to_logger=json_lines_to_logger)
                elif not dry_run_quiet:
                    if not skipped_by_policy:
                        logger.info(
                            "[dry-run] would repair %s -> *%s; CONVERTED junk: %s; small_review junk: %s; animated_ic=%s",
                            src,
                            new_ext,
                            (converted_dir / expected_jpg_name) if expected_jpg_name != "?" else "(n/a)",
                            (small_review_dir / expected_jpg_name) if expected_jpg_name != "?" else "(n/a)",
                            animated_ic,
                        )
                    elif verbose:
                        logger.info(
                            "[dry-run] skip (policy: PNG/WebP/BMP/TIFF under .jpg) %s inferred=%s; "
                            "CONVERTED junk: %s; small_review junk: %s",
                            src,
                            new_ext,
                            (converted_dir / expected_jpg_name) if expected_jpg_name != "?" else "(n/a)",
                            (small_review_dir / expected_jpg_name) if expected_jpg_name != "?" else "(n/a)",
                        )
            sk = stats.skipped_static_format_by_class.get(class_name, 0)
            if sk and not include_static_format_mismatches:
                logger.info(
                    "Class %s: %d mislabeled .jpg/.jpeg file(s) are PNG/WebP/BMP/TIFF bytes "
                    "(counted only; no repair unless --include-static-format-mismatches).",
                    class_name,
                    sk,
                )
            if not json_lines and not dry_run_quiet and not mismatches:
                logger.info("[dry-run] no mislabeled .jpg/.jpeg in %s", class_name)
            continue

        # Live repair: same scheduling as dry-run — one tree scan, one basename map, one sorted pass.
        failed_paths: Set[str] = set()
        check_cancel_event(cancel_event)
        image_files = converter.find_image_files(class_dir, log_scan=False)
        static_paths, _ = converter._split_static_and_extract(image_files)
        static_paths.sort(key=lambda x: str(x))
        basename_map = assign_still_convert_output_basenames(
            static_paths,
            output_dir=converted_dir,
        )
        mismatches = [
            p
            for p in static_paths
            if p.suffix.lower() in (".jpg", ".jpeg") and not is_jpeg_magic(p)
        ]

        for src in sorted(mismatches, key=lambda x: str(x)):
            check_cancel_event(cancel_event)
            if not src.is_file():
                continue
            key = str(src.resolve())
            if key in failed_paths:
                continue

            expected_jpg_name = basename_map.get(src)
            if not expected_jpg_name:
                stats.errors.append(f"no basename mapping for {src}")
                failed_paths.add(key)
                stats.files_failed += 1
                continue

            stats.mismatches_found += 1

            head = read_file_header(src)
            sniff = sniff_container(head)
            pil_fmt, _ = pillow_format_and_gif_info(src)
            new_ext = extension_for_container(sniff, pil_fmt)
            if not new_ext:
                logger.error("Cannot infer extension for %s (sniff=%s pil=%s)", src, sniff, pil_fmt)
                failed_paths.add(key)
                stats.files_failed += 1
                continue

            junk_converted = converted_dir / expected_jpg_name
            junk_small = small_review_dir / expected_jpg_name

            n_gif_frames = (
                _gif_frame_count(src) if (pil_fmt == "GIF" or sniff == "gif") else 1
            )
            animated_ic = (
                model_type == ModelType.IMAGE_CLASSIFICATION
                and (pil_fmt == "GIF" or sniff == "gif")
                and n_gif_frames > 1
            )

            if (
                _static_format_skipped_by_policy(new_ext, include_static_format_mismatches)
                and not animated_ic
            ):
                stats.skipped_static_format_by_class[class_name] = (
                    stats.skipped_static_format_by_class.get(class_name, 0) + 1
                )
                if verbose:
                    logger.info(
                        "[skip] policy (PNG/WebP/BMP/TIFF under .jpg): %s inferred=%s "
                        "(use --include-static-format-mismatches to repair)",
                        src,
                        new_ext,
                    )
                if json_lines and verbose:
                    skip_row: Dict[str, Any] = {
                        **_mismatch_row_base(src),
                        "class": class_name,
                        "expected_converted_basename": expected_jpg_name,
                        "inferred_extension": new_ext,
                        "animated_ic_random_frame": animated_ic,
                        "skipped_by_policy": True,
                        "dry_run": False,
                    }
                    _emit_json_line(skip_row, json_lines_to_logger=json_lines_to_logger)
                continue

            created: List[Path] = []
            target_jpeg: Optional[Path] = None
            try:
                rel_original_old = _rel_under_raw(raw_data_dir, src)
                converted_out: Path
                if animated_ic:
                    target_jpeg = converted_dir / convert_output_jpeg_filename(
                        src,
                        output_dir=converted_dir,
                        also_under_dirs=(review_dir,),
                    )
                    ok = extract_random_gif_frame_to_jpeg(src, target_jpeg, rng)
                    if not ok:
                        raise RuntimeError("extract_random_gif_frame_to_jpeg failed")
                    created.append(target_jpeg)
                    rp = _copy_visual_review_like_convert(target_jpeg, review_dir)
                    if rp:
                        created.append(rp)
                    converted_out = target_jpeg
                else:
                    from PIL import Image

                    out_still = converted_dir / expected_jpg_name
                    with Image.open(src) as im:
                        if im.format == "GIF" and getattr(im, "n_frames", 1) > 1:
                            im.seek(0)
                        rgb = im.convert("RGB")
                    if not pil_image_to_jpeg_normalized(rgb, out_still):
                        raise RuntimeError("pil_image_to_jpeg_normalized failed")
                    created.append(out_still)
                    converted_out = out_still

                dest_src = _unique_dest(src.with_suffix(new_ext))
                src.rename(dest_src)

                if animated_ic:
                    if junk_converted.exists() and junk_converted.resolve() != converted_out.resolve():
                        try:
                            junk_converted.unlink()
                        except OSError as e:
                            logger.warning("Could not remove stale CONVERTED file %s: %s", junk_converted, e)

                if junk_small.exists():
                    try:
                        junk_small.unlink()
                    except OSError as e:
                        logger.warning("Could not remove stale small review file %s: %s", junk_small, e)

                if snapshot is not None:
                    try:
                        _update_snapshot_record_after_repair(
                            snapshot,
                            raw_data_dir,
                            class_name,
                            rel_original_old,
                            dest_src,
                            converted_out,
                            logger=logger,
                        )
                        stats.snapshot_records_updated += 1
                    except Exception as e:
                        logger.warning("Snapshot update failed after repair: %s", e, exc_info=True)
                        stats.snapshot_apply_errors.append(f"{rel_original_old}: {e}")

                stats.files_repaired += 1
                stats.actionable_mismatches_found += 1
                if json_lines:
                    repaired_row: Dict[str, Any] = {
                        **_mismatch_row_base(dest_src),
                        "class": class_name,
                        "expected_converted_basename": expected_jpg_name,
                        "inferred_extension": new_ext,
                        "animated_ic_random_frame": animated_ic,
                        "skipped_by_policy": False,
                        "dry_run": False,
                        "result": "repaired",
                        "converted_jpeg": str(converted_out),
                    }
                    _emit_json_line(repaired_row, json_lines_to_logger=json_lines_to_logger)
                logger.info("Repaired -> %s", dest_src)
            except Exception as e:
                logger.exception("Repair failed for %s: %s", src, e)
                for p in created:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                failed_paths.add(key)
                stats.files_failed += 1
                stats.errors.append(f"{src}: {e}")

        if not dry_run:
            sk_live = stats.skipped_static_format_by_class.get(class_name, 0)
            if sk_live and not include_static_format_mismatches:
                logger.info(
                    "Class %s: %d mislabeled .jpg/.jpeg file(s) are PNG/WebP/BMP/TIFF bytes "
                    "(counted only; no repair unless --include-static-format-mismatches).",
                    class_name,
                    sk_live,
                )

    elapsed_sec: Optional[float] = None
    if wall_t0 is not None:
        elapsed_sec = time.perf_counter() - wall_t0

    if (
        not dry_run
        and snapshot is not None
        and repair_started_iso is not None
        and elapsed_sec is not None
    ):
        messages: List[str] = []
        messages.extend(stats.errors)
        messages.extend(stats.snapshot_apply_errors)
        messages.append(f"wall_seconds:{elapsed_sec:.6f}")
        messages.append(f"snapshot_records_updated:{stats.snapshot_records_updated}")
        set_step_errors_for_invocation(
            snapshot,
            ModelBuildStepCommand.FIX_JPEG_EXTENSION_MISMATCH.value,
            repair_started_iso,
            messages,
        )
        save_unified_snapshot(snapshot, raw_data_dir, logger=logger)

    ok = stats.files_failed == 0
    timing_tail = f" wall_seconds={elapsed_sec:.6f}" if elapsed_sec is not None else ""
    skipped_total = sum(stats.skipped_static_format_by_class.values())
    log_completion_info(
        logger,
        ok,
        f"repaired={stats.files_repaired} failed={stats.files_failed} dry_run_actions={stats.dry_run_actions} "
        f"mismatches_found={stats.mismatches_found} actionable={stats.actionable_mismatches_found} "
        f"skipped_static_format={skipped_total}{timing_tail}",
    )
    return ok, stats


def run_repair_cli_main(argv: Optional[List[str]] = None) -> int:
    """Entry for ``mb data fix-jpeg-extension-mismatch``."""
    from mb.cli import run_data_subcommand_cli

    if argv is None:
        argv = sys.argv[1:]
    return run_data_subcommand_cli("fix-jpeg-extension-mismatch", argv)


if __name__ == "__main__":
    sys.exit(run_repair_cli_main())
