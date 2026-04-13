#!/usr/bin/env python3
"""
Convert raw class-folder images to a normalized layout (JPEG-oriented).

:class:`ImageConverter` walks per-class directories under ``raw_data_dir``, validates
inputs, and writes outputs while maintaining a unified snapshot for the pipeline.

**CLI:** ``mb data convert``; ``python -m mb.data.convert`` delegates via
:func:`mb.cli.run_data_subcommand_cli`.
"""

import shutil
import sys
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict
# Image processing imports
try:
    from PIL import Image, UnidentifiedImageError
    # Increase decompression bomb limit to handle very large images
    # Default is ~178MP, we'll allow up to ~500MP (reasonable for high-res images)
    Image.MAX_IMAGE_PIXELS = 500 * 1024 * 1024  # 500MP
except ImportError:
    raise ImportError("Error: PIL/Pillow not available. Image conversion is required.")

# Import centralized logging configuration
from mb.utils.logging_setup import log_completion_info, log_startup_info, setup_logging
from mb.cancellation import check_cancel_event
from mb.utils.snapshot import (
    UnifiedSnapshot,
    calculate_file_hash,
    find_unified_snapshot,
    flatten_convert_stats_errors,
    generate_run_id,
    preload_gather_cache,
    save_unified_snapshot,
    set_step_errors_for_invocation,
)
from mb.data.class_layout import (
    CONVERTED_MEDIA_SUBDIR,
    VISUAL_MEDIA_REVIEW_SUBDIR,
    discover_class_names,
    normalize_qualifying_subdir,
    POST_CONVERT_SUBDIR_NAMES,
)
from mb.data.file_types import (
    configured_media_suffixes,
    configured_video_suffixes,
    normalized_jpeg_suffixes,
)
from mb.data.media_utils import (
    classify_convert_source,
    extract_random_gif_frame_to_jpeg,
    extract_random_video_frame_to_jpeg,
)
from mb.pipeline_config import get_pipeline_config
from mb.models.types import ModelBuildStepCommand, ModelType, VisualMediaSourceType
from mb.space_estimate import check_convert_allowed, merge_convert_estimate_into_snapshot
from mb.utils.utils import (
    assign_still_convert_output_basenames,
    convert_output_jpeg_filename,
    plain_still_jpeg_basename,
    Utils,
)

# Configure logging
logger = setup_logging(script_name="convert")

# Media suffixes: :func:`configured_media_suffixes` / :func:`normalized_jpeg_suffixes` in :mod:`mb.data.file_types`

# Default raw data directory (contains all class directories)
DEFAULT_RAW_DATA_DIR = Path("raw_data")


class ImageConverter:
    """Handles conversion of images to JPEG format for all class directories."""
    
    def __init__(self, raw_data_dir: Path, model_type: Optional[ModelType] = None):
        self.raw_data_dir = Path(raw_data_dir)
        pc = get_pipeline_config()
        self.model_type = (
            model_type
            if model_type is not None
            else ModelType.from_pipeline_value(pc.get("model.default_type"))
        )
        
        # Statistics tracking
        self.stats = {
            'total_files_found': 0,
            'files_converted': 0,
            'files_copied': 0,
            'files_visual_extracted': 0,
            'files_promoted_to_plain': 0,
            'files_skipped': 0,
            'errors': defaultdict(list),
        }
        
        # Ensure raw data directory exists
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified snapshot (will be created or loaded)
        self.unified_snapshot: Optional[UnifiedSnapshot] = None
        self.run_id: Optional[str] = None
        self._requested_run_id: Optional[str] = None
        self._convert_step_started_at: Optional[str] = None
        self._rng: random.Random = random.Random()

    def _scan_suffixes(self) -> List[str]:
        exts = set(configured_media_suffixes())
        if self.model_type == ModelType.IMAGE_CLASSIFICATION:
            exts |= set(configured_video_suffixes())
        return sorted(exts)

    def _split_static_and_extract(self, paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """Split inputs into normal still-image conversion vs random-frame extraction."""
        static: List[Path] = []
        extract: List[Path] = []
        for p in paths:
            st = classify_convert_source(p, model_type=self.model_type)
            if st in (VisualMediaSourceType.VIDEO_EXTRACT, VisualMediaSourceType.ANIMATED_GIF_EXTRACT):
                extract.append(p)
            else:
                static.append(p)
        return static, extract

    def validate_configuration(self) -> bool:
        """Validate the raw data directory."""
        # Wake-aware directory check helps avoid false negatives on sleeping external drives.
        if not Utils.isdir_with_retry(str(self.raw_data_dir)):
            logger.error(f"Raw data path is not a directory: {self.raw_data_dir}")
            return False
            
        # Check if raw data directory is writable
        try:
            test_file = self.raw_data_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            logger.error(f"Cannot write to raw data directory {self.raw_data_dir}: {e}")
            return False
            
        return True
    
    def find_image_files(self, class_dir: Path, *, log_scan: bool = True) -> List[Path]:
        """
        Find all image files in a class directory, excluding post-convert output trees
        (``CONVERTED``, legacy ``JPEG_IMAGES``).
        
        Args:
            class_dir: Class directory to scan (e.g. ``raw_data/<class_name>``)
            log_scan: If false, skip INFO lines about scanning (for callers that rescan often).
            
        Returns:
            List of image file paths found
        """
        image_files = []
        post_convert_roots = [class_dir / n for n in POST_CONVERT_SUBDIR_NAMES]

        def _under_post_convert(p: Path) -> bool:
            for root in post_convert_roots:
                if root.exists() and p.is_relative_to(root):
                    return True
            return False

        try:
            # Check if directory has subdirectories (excluding post-convert outputs)
            subdirs = [
                d
                for d in class_dir.iterdir()
                if d.is_dir() and d.name not in POST_CONVERT_SUBDIR_NAMES
            ]

            if subdirs:
                # Has subdirectories: scan recursively in subdirectories (not root)
                if log_scan:
                    logger.info(f"Scanning subdirectories in: {class_dir.name}")
                for subdir in subdirs:
                    logger.debug(f"  Scanning subdirectory: {subdir.name}")
                    for ext in self._scan_suffixes():
                        for file_path in subdir.rglob(f'*{ext}'):
                            image_files.append(file_path)
            else:
                # No subdirectories: scan root level (excluding post-convert dirs if present)
                if log_scan:
                    logger.info(f"Scanning root level in: {class_dir.name} (no subdirectories found)")
                for ext in self._scan_suffixes():
                    for file_path in class_dir.glob(f'*{ext}'):
                        if _under_post_convert(file_path):
                            continue
                        image_files.append(file_path)
                        
        except PermissionError as e:
            logger.error(f"Permission denied accessing {class_dir}: {e}")
            self.stats['errors']['permission'].append(str(class_dir))
        except Exception as e:
            logger.error(f"Error scanning {class_dir}: {e}")
            self.stats['errors']['scanning'].append((str(class_dir), str(e)))
        
        return image_files
    
    
    def convert_to_jpeg(self, source_path: Path, target_path: Path) -> bool:
        """Convert image to JPEG format."""
        try:
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source_path}")
                return False
                
            if not source_path.is_file():
                logger.error(f"Source path is not a file: {source_path}")
                return False
                
            with Image.open(source_path) as img:
                # Convert to RGB if necessary (for RGBA images)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if any dimension exceeds 4000px
                width, height = img.size
                max_dimension = max(width, height)
                if max_dimension > 4000:
                    # Calculate scaling factor to bring max dimension to 4000px
                    scale_factor = 4000 / max_dimension
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    logger.debug(f"Resized {source_path.name} from {width}x{height} to {new_width}x{new_height}")
                
                # Save as JPEG
                img.save(target_path, 'JPEG', quality=95, optimize=True)
                
                # Verify the saved file exists and has content
                if not target_path.exists() or target_path.stat().st_size == 0:
                    logger.error(f"Failed to save converted image: {target_path}")
                    return False
                    
                return True
                
        except UnidentifiedImageError:
            logger.error(f"Cannot identify image format: {source_path}")
            return False
        except OSError as e:
            error_msg = str(e)
            if "decompression bomb" in error_msg.lower() or "exceeds limit" in error_msg.lower():
                logger.warning(f"Image too large (decompression bomb protection): {source_path}")
                logger.warning(f"  Error: {error_msg}")
                logger.warning(f"  Consider increasing Image.MAX_IMAGE_PIXELS or skipping this file")
            else:
                logger.error(f"OS error converting {source_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error converting {source_path}: {e}")
            return False
    
    def copy_jpeg_file(self, source_path: Path, target_path: Path) -> bool:
        """Copy and resize a JPEG file to the target directory if needed."""
        try:
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source_path}")
                return False
            
            # Open image to check dimensions and resize if needed
            with Image.open(source_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if any dimension exceeds 4000px
                width, height = img.size
                max_dimension = max(width, height)
                if max_dimension > 4000:
                    # Calculate scaling factor to bring max dimension to 4000px
                    scale_factor = 4000 / max_dimension
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    logger.debug(f"Resized {source_path.name} from {width}x{height} to {new_width}x{new_height}")
                    
                    # Save resized image
                    img.save(target_path, 'JPEG', quality=95, optimize=True)
                else:
                    # No resizing needed, just copy
                    shutil.copy2(source_path, target_path)
            
            # Verify the file exists and has content
            if not target_path.exists() or target_path.stat().st_size == 0:
                logger.error(f"Failed to process file: {target_path}")
                return False
            
            return True
                
        except PermissionError as e:
            logger.error(f"Permission error copying {source_path}: {e}")
            self.stats['errors']['permission'].append(str(source_path))
            return False
        except OSError as e:
            error_msg = str(e)
            if "decompression bomb" in error_msg.lower() or "exceeds limit" in error_msg.lower():
                logger.warning(f"Image too large (decompression bomb protection): {source_path}")
                logger.warning(f"  Error: {error_msg}")
                self.stats['errors']['decompression_bomb'].append((str(source_path), str(e)))
            else:
                logger.error(f"OS error copying {source_path}: {e}")
                self.stats['errors']['copy'].append((str(source_path), str(e)))
            return False
        except Exception as e:
            logger.error(f"Unexpected error copying {source_path}: {e}")
            self.stats['errors']['unexpected'].append((str(source_path), str(e)))
            return False

    def _sync_post_conversion_still(
        self, class_dir: Path, source_path: Path, target_path: Path
    ) -> None:
        """Update snapshot ``converted`` for a still image so dataset/train can match by path/MD5."""
        if not self.unified_snapshot:
            return
        original_hash = calculate_file_hash(
            source_path,
            algorithm="md5",
            raw_data_dir=self.raw_data_dir,
            logger=logger,
        )
        if not original_hash:
            return
        conv_md5 = calculate_file_hash(
            target_path,
            algorithm="md5",
            raw_data_dir=self.raw_data_dir,
            logger=logger,
        )
        conv_sha = calculate_file_hash(
            target_path,
            algorithm="sha256",
            raw_data_dir=self.raw_data_dir,
            logger=logger,
        )
        if not conv_md5 or not conv_sha:
            return
        try:
            rel = str(target_path.relative_to(self.raw_data_dir))
        except ValueError:
            rel = str(target_path)
        self.unified_snapshot.add_post_conversion_image(
            class_name=class_dir.name,
            converted_path=rel,
            converted_basename=target_path.name,
            converted_md5=conv_md5,
            converted_sha256=conv_sha,
            original_info={"original_hash": original_hash},
        )

    @staticmethod
    def _still_converted_nonempty(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _resolve_still_skip_or_promote_legacy(
        self,
        source_path: Path,
        target_filename: str,
        output_dir: Path,
    ) -> Optional[Tuple[Path, bool]]:
        """
        If the assigned output already exists, return ``(path, False)``.

        If the assigned name is the plain ``{stem}.jpg`` and only a prior hash-suffixed file exists
        (older converter output), rename it to the plain name when possible and return
        ``(plain_path, True)``.
        """
        target_path = output_dir / target_filename
        if self._still_converted_nonempty(target_path):
            return (target_path, False)
        try:
            expected_plain = plain_still_jpeg_basename(source_path.stem)
        except ValueError:
            return None
        if target_filename != expected_plain:
            return None
        legacy_name = convert_output_jpeg_filename(source_path, output_dir=output_dir)
        if legacy_name == target_filename:
            return None
        legacy_path = output_dir / legacy_name
        if not self._still_converted_nonempty(legacy_path):
            return None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            legacy_path.replace(target_path)
        except OSError:
            return (legacy_path, False)
        return (target_path, True)

    def process_files(self, image_files: List[Path], class_dir: Path) -> None:
        """
        Process the image files for a class directory, converting to JPEG or copying if already JPEG.
        Normalized outputs go under ``CONVERTED_MEDIA_SUBDIR`` (``CONVERTED``).
        
        Args:
            image_files: List of image file paths to process
            class_dir: Class directory (determines where to place output)
        """
        output_dir = class_dir / CONVERTED_MEDIA_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {len(image_files)} files for {class_dir.name} (output: {CONVERTED_MEDIA_SUBDIR}/)")

        basename_by_source = assign_still_convert_output_basenames(
            image_files, output_dir=output_dir
        )

        for i, source_path in enumerate(image_files, 1):
            if i % 100 == 0:
                check_cancel_event(getattr(self, "_cancel_event", None))
                logger.info(f"Progress: {i}/{len(image_files)} files processed for {class_dir.name}")
            
            # Determine if file is already JPEG
            is_jpeg = source_path.suffix.lower() in normalized_jpeg_suffixes()
            
            target_filename = basename_by_source[source_path]
            target_path = output_dir / target_filename

            skip_t = self._resolve_still_skip_or_promote_legacy(
                source_path, target_filename, output_dir
            )
            if skip_t is not None:
                skip_existing, promoted = skip_t
                logger.debug(
                    f"Skipping {source_path.name} — output present in {CONVERTED_MEDIA_SUBDIR}/: "
                    f"{skip_existing.name}"
                )
                self._sync_post_conversion_still(class_dir, source_path, skip_existing)
                self.stats['files_skipped'] += 1
                if promoted:
                    self.stats['files_promoted_to_plain'] += 1
                continue

            # Process the file
            if is_jpeg:
                # Already JPEG - just copy
                if self.copy_jpeg_file(source_path, target_path):
                    self.stats['files_copied'] += 1
                    self._sync_post_conversion_still(class_dir, source_path, target_path)
                    logger.debug(f"Copied JPEG: {source_path.name} -> {target_path.name}")
                else:
                    self.stats['files_skipped'] += 1
            else:
                # Need to convert to JPEG
                if self.convert_to_jpeg(source_path, target_path):
                    self.stats['files_converted'] += 1
                    self._sync_post_conversion_still(class_dir, source_path, target_path)
                    logger.debug(f"Converted: {source_path.name} ({source_path.suffix}) -> {target_path.name}")
                else:
                    self.stats['files_skipped'] += 1
                    self.stats['errors']['conversion'].append(str(source_path))

    def process_visual_extractions(self, extract_paths: List[Path], class_dir: Path) -> None:
        """
        For videos and animated GIFs, extract one random frame as JPEG into ``CONVERTED``
        and duplicate into ``VISUAL_MEDIA_REVIEW_SUBDIR`` for review.
        """
        if not extract_paths:
            return
        output_dir = class_dir / CONVERTED_MEDIA_SUBDIR
        review_dir = class_dir / VISUAL_MEDIA_REVIEW_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Extracting random frames for {len(extract_paths)} visual media file(s) "
            f"into {CONVERTED_MEDIA_SUBDIR}/ and {VISUAL_MEDIA_REVIEW_SUBDIR}/"
        )

        for i, source_path in enumerate(extract_paths, 1):
            if i % 50 == 0:
                check_cancel_event(getattr(self, "_cancel_event", None))
                logger.info(f"Visual extraction progress: {i}/{len(extract_paths)} for {class_dir.name}")

            target_filename = convert_output_jpeg_filename(
                source_path,
                output_dir=output_dir,
                also_under_dirs=(review_dir,),
            )
            target_path = output_dir / target_filename

            try:
                if target_path.stat().st_size > 0:
                    logger.debug(
                        f"Skipping visual extract for {source_path.name} — output exists: {target_path.name}"
                    )
                    self.stats["files_skipped"] += 1
                    continue
            except (OSError, FileNotFoundError):
                pass

            ok = False
            if source_path.suffix.lower() in configured_video_suffixes():
                ok = extract_random_video_frame_to_jpeg(source_path, target_path, self._rng)
                if not ok:
                    self.stats["errors"]["visual_video"].append(str(source_path))
                    logger.error(
                        f"Could not extract a frame from video (install imageio + imageio-ffmpeg?): {source_path}"
                    )
            elif source_path.suffix.lower() == ".gif":
                ok = extract_random_gif_frame_to_jpeg(source_path, target_path, self._rng)
                if not ok:
                    self.stats["errors"]["visual_gif"].append(str(source_path))
                    logger.error(f"Could not extract a frame from GIF: {source_path}")
            else:
                self.stats["errors"]["visual_unknown"].append(str(source_path))
                continue

            if ok:
                self.stats["files_visual_extracted"] += 1
                review_path = review_dir / target_path.name
                r_counter = 1
                orig_review_stem = review_path.stem
                while review_path.exists():
                    review_path = review_dir / f"{orig_review_stem}_{r_counter}.jpg"
                    r_counter += 1
                try:
                    shutil.copy2(target_path, review_path)
                except OSError as e:
                    logger.warning(f"Could not copy review JPEG {review_path}: {e}")
                if self.unified_snapshot:
                    self.unified_snapshot.add_pre_conversion_image(target_path, self.raw_data_dir)
                logger.debug(f"Visual extract: {source_path.name} -> {target_path.name}")
            else:
                self.stats["files_skipped"] += 1

    def print_summary(self) -> None:
        """Print a summary of the operation."""
        logger.info("=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Total files found: {self.stats['total_files_found']}")
        logger.info(f"Files converted to JPEG: {self.stats['files_converted']}")
        logger.info(f"JPEG files copied: {self.stats['files_copied']}")
        if self.stats["files_visual_extracted"]:
            logger.info(f"Random frames from video/GIF (JPEG): {self.stats['files_visual_extracted']}")
        logger.info(f"Files skipped (errors): {self.stats['files_skipped']}")
        
        if self.stats['errors']:
            logger.info("\nERRORS ENCOUNTERED:")
            max_detail = 100
            for error_type, errors in self.stats['errors'].items():
                logger.info(f"  {error_type}: {len(errors)} errors")
                for error in errors[:max_detail]:
                    logger.info(f"    - {error}")
                if len(errors) > max_detail:
                    logger.info(f"    ... and {len(errors) - max_detail} more")
        
        if self.stats["files_promoted_to_plain"]:
            logger.info(
                "Legacy hash-named JPEGs renamed to plain stem.jpg: "
                f"{self.stats['files_promoted_to_plain']}"
            )
        total_processed = (
            self.stats["files_converted"]
            + self.stats["files_copied"]
            + self.stats["files_visual_extracted"]
            + self.stats["files_promoted_to_plain"]
        )
        logger.info(f"\nTotal files processed: {total_processed}")
    
    def run(
        self,
        cancel_event: Optional[threading.Event] = None,
        *,
        skip_space_check: bool = False,
        run_id: Optional[str] = None,
    ) -> bool:
        """Main execution method. *cancel_event* is checked between classes and during file loops.

        If *run_id* is set, loads ``snapshot_<run_id>.json`` under ``raw_data_dir`` and updates
        that unified snapshot in place. If omitted, generates a new run ID and snapshot file.
        """
        self._cancel_event = cancel_event
        self._skip_space_check = skip_space_check
        self._requested_run_id = (str(run_id).strip() or None) if run_id is not None else None
        try:
            return self._run_impl()
        finally:
            self._cancel_event = None
            self._requested_run_id = None

    def _run_impl(self) -> bool:
        log_startup_info(logger, "Image to JPEG conversion process")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        
        check_cancel_event(self._cancel_event)
        
        # Validate configuration
        if not self.validate_configuration():
            return False

        resume_rid = getattr(self, "_requested_run_id", None)
        snapshot_for_check: Optional[UnifiedSnapshot] = None
        if resume_rid:
            loaded = find_unified_snapshot(
                [self.raw_data_dir], run_id=resume_rid, logger=logger
            )
            if loaded is None:
                logger.error(
                    "No unified snapshot found for run_id %r under %s "
                    "(expected snapshot_%s.json).",
                    resume_rid,
                    self.raw_data_dir,
                    resume_rid,
                )
                return False
            snapshot_for_check = loaded
            logger.info(
                "Loaded unified snapshot for run_id %s — will update the same snapshot file",
                loaded.run_id,
            )

        allowed, space_report = check_convert_allowed(
            self.raw_data_dir,
            self.model_type,
            snapshot=snapshot_for_check,
            skip_space_check=getattr(self, "_skip_space_check", False),
        )
        if not allowed:
            logger.error(
                "Insufficient disk space for convert (heuristic). "
                "Free space or use skip_space_check / --skip-space-check if you accept the risk."
            )
            return False
        self._last_space_report = space_report

        if resume_rid:
            self.unified_snapshot = snapshot_for_check
            self.run_id = self.unified_snapshot.run_id
        else:
            self.run_id = generate_run_id()
            logger.info(f"Creating new unified snapshot with run_id: {self.run_id}")
            self.unified_snapshot = UnifiedSnapshot(
                run_id=self.run_id,
                raw_data_dir=str(self.raw_data_dir),
            )
        merge_convert_estimate_into_snapshot(self.unified_snapshot, self._last_space_report)
        self._convert_step_started_at = datetime.now(timezone.utc).isoformat()

        # Preload gather cache for faster hash lookups
        cache_loaded = preload_gather_cache(self.raw_data_dir)
        if cache_loaded:
            logger.info("Gather cache loaded successfully - hash lookups will be faster")
        
        pc = get_pipeline_config()
        qual = normalize_qualifying_subdir(pc.get("data.class_qualifying_subdir"))
        ex = pc.get("data.class_names")
        explicit_list = ex if isinstance(ex, list) else None
        class_names = discover_class_names(
            self.raw_data_dir,
            explicit=explicit_list,
            class_qualifying_subdir=qual,
        )
        if not class_names:
            logger.warning("No class directories found under %s", self.raw_data_dir)
            return True

        # Process each class directory
        all_image_files = []
        for class_name in class_names:
            check_cancel_event(self._cancel_event)
            class_dir = self.raw_data_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory does not exist: {class_dir}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing class: {class_name}")
            logger.info(f"{'='*60}")
            
            # Find image files for this class (excluding post-convert output trees)
            image_files = self.find_image_files(class_dir)
            static_paths, extract_paths = self._split_static_and_extract(image_files)

            if not image_files:
                logger.info(f"No image files found in {class_name}")
                continue

            self.stats["total_files_found"] += len(image_files)
            logger.info(
                f"Found {len(image_files)} media file(s) in {class_name} "
                f"({len(static_paths)} still image(s), {len(extract_paths)} video/animated-GIF)"
            )

            # Pre-conversion snapshot: still inputs only (video/GIF sources are hashed after frame JPEG exists)
            logger.info(f"Adding pre-conversion images to snapshot for {class_name}...")
            for image_file in static_paths:
                self.unified_snapshot.add_pre_conversion_image(image_file, self.raw_data_dir)

            self.process_files(static_paths, class_dir)
            self.process_visual_extractions(extract_paths, class_dir)
            all_image_files.extend(image_files)
        
        if not all_image_files:
            logger.warning("No image files found in any class directory")
            return True  # Not an error, just nothing to do
        
        # Save unified snapshot at raw_data level (single file for all classes)
        if self.unified_snapshot and self._convert_step_started_at:
            set_step_errors_for_invocation(
                self.unified_snapshot,
                ModelBuildStepCommand.CONVERT.value,
                self._convert_step_started_at,
                flatten_convert_stats_errors(self.stats.get("errors")),
            )
        snapshot_path = save_unified_snapshot(self.unified_snapshot, self.raw_data_dir, logger)
        logger.info(f"Run ID for this pipeline: {self.run_id}")
        logger.info(
            "Subsequent jobs should use this run ID to update the same snapshot file"
        )
        
        # Print summary
        self.print_summary()
        
        # Log completion
        success = (
            self.stats["files_converted"]
            + self.stats["files_copied"]
            + self.stats["files_visual_extracted"]
            + self.stats["files_promoted_to_plain"]
        ) > 0
        message = (
            f"Converted {self.stats['files_converted']} files, copied {self.stats['files_copied']} JPEG files, "
            f"extracted {self.stats['files_visual_extracted']} frame(s) from video/GIF, "
            f"promoted {self.stats['files_promoted_to_plain']} legacy hash-named JPEG(s) to plain stem.jpg"
        )
        log_completion_info(logger, success, message)
        
        return success


if __name__ == "__main__":
    from mb.cli import run_data_subcommand_cli

    sys.exit(run_data_subcommand_cli("convert"))

