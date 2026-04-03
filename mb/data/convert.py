#!/usr/bin/env python3
"""
Convert raw class-folder images to a normalized layout (JPEG-oriented).

:class:`ImageConverter` walks per-class directories under ``raw_data_dir``, validates
inputs, and writes outputs while maintaining a unified snapshot for the pipeline.

**CLI:** ``mb data convert``; ``python -m mb.data.convert`` delegates via
:func:`mb.cli.run_data_subcommand_cli`.
"""

import sys
import shutil
import threading
from pathlib import Path
from typing import List, Set, Optional
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
    UnifiedSnapshot, generate_run_id, save_unified_snapshot, preload_gather_cache
)
from mb.data.class_layout import discover_class_names, normalize_qualifying_subdir
from mb.pipeline_config import get_pipeline_config

# Configure logging
logger = setup_logging(script_name="convert_to_jpeg")

# Image file extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
JPEG_EXTENSIONS = {'.jpg', '.jpeg'}

# Default raw data directory (contains all class directories)
DEFAULT_RAW_DATA_DIR = Path("raw_data")
JPEG_IMAGES_DIR = "JPEG_IMAGES"  # Output subdirectory for converted JPEGs per class


class ImageConverter:
    """Handles conversion of images to JPEG format for all class directories."""
    
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        
        # Statistics tracking
        self.stats = {
            'total_files_found': 0,
            'files_converted': 0,
            'files_copied': 0,
            'files_skipped': 0,
            'errors': defaultdict(list),
        }
        
        # Ensure raw data directory exists
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified snapshot (will be created or loaded)
        self.unified_snapshot: Optional[UnifiedSnapshot] = None
        self.run_id: Optional[str] = None
    
    def validate_configuration(self) -> bool:
        """Validate the raw data directory."""
        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory does not exist: {self.raw_data_dir}")
            return False
            
        if not self.raw_data_dir.is_dir():
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
    
    def find_image_files(self, class_dir: Path) -> List[Path]:
        """
        Find all image files in a class directory, excluding JPEG_IMAGES subdirectory.
        
        Args:
            class_dir: Class directory to scan (e.g., raw_data/coherent)
            
        Returns:
            List of image file paths found
        """
        image_files = []
        jpeg_images_dir = class_dir / JPEG_IMAGES_DIR
        
        try:
            # Check if directory has subdirectories (excluding JPEG_IMAGES)
            subdirs = [d for d in class_dir.iterdir() if d.is_dir() and d != jpeg_images_dir]
            
            if subdirs:
                # Has subdirectories: scan recursively in subdirectories (not root)
                logger.info(f"Scanning subdirectories in: {class_dir.name}")
                for subdir in subdirs:
                    logger.debug(f"  Scanning subdirectory: {subdir.name}")
                    for ext in IMAGE_EXTENSIONS:
                        for file_path in subdir.rglob(f'*{ext}'):
                            image_files.append(file_path)
            else:
                # No subdirectories: scan root level (excluding JPEG_IMAGES if it exists)
                logger.info(f"Scanning root level in: {class_dir.name} (no subdirectories found)")
                # Check once if JPEG_IMAGES exists (performance optimization)
                jpeg_images_exists = jpeg_images_dir.exists()
                for ext in IMAGE_EXTENSIONS:
                    for file_path in class_dir.glob(f'*{ext}'):
                        # Exclude JPEG_IMAGES directory contents
                        if jpeg_images_exists and file_path.is_relative_to(jpeg_images_dir):
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
    
    def process_files(self, image_files: List[Path], class_dir: Path) -> None:
        """
        Process the image files for a class directory, converting to JPEG or copying if already JPEG.
        All converted JPEGs are placed in JPEG_IMAGES subdirectory for consistency.
        
        Args:
            image_files: List of image file paths to process
            class_dir: Class directory (determines where to place output)
        """
        # Always place converted JPEGs in JPEG_IMAGES subdirectory
        output_dir = class_dir / JPEG_IMAGES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {len(image_files)} files for {class_dir.name} (output: {JPEG_IMAGES_DIR}/)")
        
        for i, source_path in enumerate(image_files, 1):
            if i % 100 == 0:
                check_cancel_event(getattr(self, "_cancel_event", None))
                logger.info(f"Progress: {i}/{len(image_files)} files processed for {class_dir.name}")
            
            # Determine if file is already JPEG
            is_jpeg = source_path.suffix.lower() in JPEG_EXTENSIONS
            
            # Create target filename - always use .jpg extension
            target_filename = f"{source_path.stem}.jpg"
            target_path = output_dir / target_filename
            
            # Check if file already exists in JPEG_IMAGES (skip if already converted)
            try:
                if target_path.stat().st_size > 0:
                    logger.debug(f"Skipping {source_path.name} - already exists in {JPEG_IMAGES_DIR}/: {target_path.name}")
                    self.stats['files_skipped'] += 1
                    continue
            except (OSError, FileNotFoundError):
                # File doesn't exist, proceed with conversion
                pass
            
            # Handle filename conflicts (for edge cases where filename differs)
            counter = 1
            original_target_path = target_path
            while target_path.exists():
                stem = original_target_path.stem
                target_path = output_dir / f"{stem}_{counter}.jpg"
                counter += 1
            
            # Process the file
            if is_jpeg:
                # Already JPEG - just copy
                if self.copy_jpeg_file(source_path, target_path):
                    self.stats['files_copied'] += 1
                    logger.debug(f"Copied JPEG: {source_path.name} -> {target_path.name}")
                else:
                    self.stats['files_skipped'] += 1
            else:
                # Need to convert to JPEG
                if self.convert_to_jpeg(source_path, target_path):
                    self.stats['files_converted'] += 1
                    logger.debug(f"Converted: {source_path.name} ({source_path.suffix}) -> {target_path.name}")
                else:
                    self.stats['files_skipped'] += 1
                    self.stats['errors']['conversion'].append(str(source_path))
    
    def print_summary(self) -> None:
        """Print a summary of the operation."""
        logger.info("=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Total files found: {self.stats['total_files_found']}")
        logger.info(f"Files converted to JPEG: {self.stats['files_converted']}")
        logger.info(f"JPEG files copied: {self.stats['files_copied']}")
        logger.info(f"Files skipped (errors): {self.stats['files_skipped']}")
        
        if self.stats['errors']:
            logger.info("\nERRORS ENCOUNTERED:")
            for error_type, errors in self.stats['errors'].items():
                logger.info(f"  {error_type}: {len(errors)} errors")
                for error in errors[:5]:  # Show first 5 errors
                    logger.info(f"    - {error}")
                if len(errors) > 5:
                    logger.info(f"    ... and {len(errors) - 5} more")
        
        total_processed = self.stats['files_converted'] + self.stats['files_copied']
        logger.info(f"\nTotal files processed: {total_processed}")
    
    def run(self, cancel_event: Optional[threading.Event] = None) -> bool:
        """Main execution method. *cancel_event* is checked between classes and during file loops."""
        self._cancel_event = cancel_event
        try:
            return self._run_impl()
        finally:
            self._cancel_event = None

    def _run_impl(self) -> bool:
        log_startup_info(logger, "Image to JPEG conversion process")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        
        check_cancel_event(self._cancel_event)
        
        # Validate configuration
        if not self.validate_configuration():
            return False
        
        # Create new unified snapshot with new run ID
        # This is the first script in the pipeline, so always create a new snapshot
        # Snapshot is stored at raw_data level, not per-class
        self.run_id = generate_run_id()
        logger.info(f"Creating new unified snapshot with run_id: {self.run_id}")
        self.unified_snapshot = UnifiedSnapshot(
            run_id=self.run_id,
            raw_data_dir=str(self.raw_data_dir)
        )
        
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
            
            # Find image files for this class (excluding JPEG_IMAGES)
            image_files = self.find_image_files(class_dir)
            
            if not image_files:
                logger.info(f"No image files found in {class_name}")
                continue
            
            self.stats['total_files_found'] += len(image_files)
            logger.info(f"Found {len(image_files)} image files in {class_name}")
            
            # Add pre-conversion images to snapshot
            logger.info(f"Adding pre-conversion images to snapshot for {class_name}...")
            for image_file in image_files:
                self.unified_snapshot.add_pre_conversion_image(image_file, self.raw_data_dir)
            
            # Process the files for this class
            self.process_files(image_files, class_dir)
            all_image_files.extend(image_files)
        
        if not all_image_files:
            logger.warning("No image files found in any class directory")
            return True  # Not an error, just nothing to do
        
        # Save unified snapshot at raw_data level (single file for all classes)
        snapshot_path = save_unified_snapshot(self.unified_snapshot, self.raw_data_dir, logger)
        logger.info(f"IMPORTANT: Run ID for this pipeline: {self.run_id}")
        logger.info(f"Subsequent scripts should use this run ID to update the same snapshot file")
        
        # Print summary
        self.print_summary()
        
        # Log completion
        success = self.stats['files_converted'] + self.stats['files_copied'] > 0
        message = f"Converted {self.stats['files_converted']} files and copied {self.stats['files_copied']} JPEG files"
        log_completion_info(logger, success, message)
        
        return success


if __name__ == "__main__":
    from mb.cli import run_data_subcommand_cli

    sys.exit(run_data_subcommand_cli("convert"))

