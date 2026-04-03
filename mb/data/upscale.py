#!/usr/bin/env python3
"""
Upscale undersized images staged in the small-image review area.

:class:`ImageUpscaler` enforces a minimum edge length and writes outputs next to the
review tree. Used after deduplication when very small files are quarantined.

**CLI:** ``mb data upscale``; ``python -m mb.data.upscale`` delegates via
:func:`mb.cli.run_data_subcommand_cli`.
"""

import sys
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
# Image processing imports
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise ImportError("Error: PIL/Pillow not available. Image upscaling is required.")

# Import centralized logging configuration
from mb.utils.logging_setup import log_completion_info, log_startup_info, setup_logging
from mb.cancellation import check_cancel_event
from mb.data.class_layout import discover_review_bucket_names, layout_dict_for_discovery

# Configure logging
logger = setup_logging(script_name="upscale_small_images")

# Image file extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

# Minimum dimension target
MIN_DIMENSION_TARGET = 300

# Default review directory
DEFAULT_REVIEW_DIR = Path("raw_data/small_images_review")


class ImageUpscaler:
    """Handles upscaling of small images from the review directory."""
    
    def __init__(self, review_dir: Path):
        self.review_dir = Path(review_dir)
        self.upscaled_dir = self.review_dir / "upscaled_small_images"
        self._bucket_names: List[str] = []

        # Statistics tracking
        self.stats = {
            'total_found': defaultdict(int),
            'upscaled': defaultdict(int),
            'skipped': defaultdict(int),
            'errors': defaultdict(list),
        }
        
        # Ensure upscaled directory exists
        self.upscaled_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_configuration(self) -> bool:
        """Validate the review directory."""
        if not self.review_dir.exists():
            logger.error(f"Review directory does not exist: {self.review_dir}")
            return False
            
        if not self.review_dir.is_dir():
            logger.error(f"Review path is not a directory: {self.review_dir}")
            return False
            
        return True
    
    def find_image_files(self, category: str) -> List[Path]:
        """Find all image files in a category directory."""
        category_dir = self.review_dir / category
        
        if not category_dir.exists():
            logger.debug(f"Category directory does not exist: {category_dir}")
            return []
        
        image_files = []
        
        # Use extension-specific globs to avoid expensive .is_file() checks
        for ext in IMAGE_EXTENSIONS:
            for file_path in category_dir.rglob(f'*{ext}'):
                # No .is_file() check needed - rglob with pattern only returns files
                image_files.append(file_path)
        
        return image_files
    
    def upscale_image(self, source_path: Path, target_path: Path) -> Tuple[bool, bool]:
        """
        Upscale an image so that the minimum dimension is at least MIN_DIMENSION_TARGET.
        Maintains aspect ratio.
        
        Args:
            source_path: Path to source image
            target_path: Path to save upscaled image
            
        Returns:
            Tuple of (success: bool, was_upscaled: bool)
            success: True if operation succeeded, False otherwise
            was_upscaled: True if image was actually upscaled, False if just copied
        """
        try:
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source_path}")
                return False, False
            
            # Open image and get dimensions, ensuring proper cleanup on Windows
            img = None
            width = height = 0
            try:
                img = Image.open(source_path)
                # Force PIL to read image data to avoid Windows file locking issues
                img.load()
                width, height = img.size
            finally:
                # Explicitly close the image to release file handle on Windows
                if img:
                    img.close()
            
            min_dimension = min(width, height)
            
            # Only upscale if minimum dimension is less than target
            if min_dimension >= MIN_DIMENSION_TARGET:
                # No upscaling needed, just copy
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Small delay to ensure file handle is released on Windows
                time.sleep(0.01)
                shutil.copy2(source_path, target_path)
                return True, False  # Success, but not upscaled
            
            # Calculate new dimensions maintaining aspect ratio
            scale_factor = MIN_DIMENSION_TARGET / min_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Open image again for processing
            img = None
            try:
                img = Image.open(source_path)
                img.load()
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Ensure target directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Determine output format from extension
                output_format = 'JPEG'
                if target_path.suffix.lower() in {'.png'}:
                    output_format = 'PNG'
                elif target_path.suffix.lower() in {'.webp'}:
                    output_format = 'WEBP'
                
                # Save upscaled image
                if output_format == 'JPEG':
                    img_resized.save(target_path, 'JPEG', quality=95, optimize=True)
                else:
                    img_resized.save(target_path, output_format, optimize=True)
                
                # Verify the saved file exists and has content
                if not target_path.exists() or target_path.stat().st_size == 0:
                    logger.error(f"Failed to save upscaled image: {target_path}")
                    return False, False
                
                logger.debug(f"Upscaled {source_path.name} from {width}x{height} to {new_width}x{new_height}")
                return True, True  # Success and upscaled
                
            finally:
                # Explicitly close the image to release file handle on Windows
                if img:
                    img.close()
                # Small delay to ensure file handle is released on Windows
                time.sleep(0.01)
                
        except UnidentifiedImageError:
            logger.error(f"Cannot identify image format: {source_path}")
            return False, False
        except OSError as e:
            logger.error(f"OS error upscaling {source_path}: {e}")
            return False, False
        except Exception as e:
            logger.error(f"Error upscaling {source_path}: {e}")
            return False, False
    
    def process_category(self, category: str) -> None:
        """Process all images in a category directory."""
        logger.info(f"Processing category: {category}")
        
        category_dir = self.review_dir / category
        if not category_dir.exists():
            logger.warning(f"Category directory does not exist: {category_dir}")
            return
        
        # Find all images in this category (including subdirectories)
        image_files = self.find_image_files(category)
        self.stats['total_found'][category] = len(image_files)
        
        if not image_files:
            logger.info(f"No images found in category: {category}")
            return
        
        logger.info(f"Found {len(image_files)} images in category: {category}")
        
        # Target directory for this category (flat structure, no subdirs)
        target_category_dir = self.upscaled_dir / category
        target_category_dir.mkdir(parents=True, exist_ok=True)
        
        upscaled_count = 0
        skipped_count = 0
        
        for i, source_path in enumerate(image_files, 1):
            if i % 100 == 0:
                check_cancel_event(getattr(self, "_cancel_event", None))
                logger.info(f"Progress: {i}/{len(image_files)} images processed for {category}")
            
            # Create target filename (preserve original name)
            target_filename = source_path.name
            target_path = target_category_dir / target_filename
            
            # Handle filename conflicts
            counter = 1
            original_target = target_path
            while target_path.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_path = original_target.parent / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Upscale the image
            success, was_upscaled = self.upscale_image(source_path, target_path)
            if success:
                if was_upscaled:
                    upscaled_count += 1
                else:
                    skipped_count += 1
            else:
                self.stats['errors'][category].append(str(source_path))
                skipped_count += 1
        
        self.stats['upscaled'][category] = upscaled_count
        self.stats['skipped'][category] = skipped_count
        
        logger.info(f"Category {category}: {upscaled_count} upscaled, {skipped_count} skipped")
    
    def print_summary(self) -> None:
        """Print a summary of the upscaling operation."""
        logger.info("=" * 60)
        logger.info("UPSCALING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Review directory: {self.review_dir}")
        logger.info(f"Upscaled directory: {self.upscaled_dir}")
        logger.info(f"Target minimum dimension: {MIN_DIMENSION_TARGET}px")
        
        total_found = sum(self.stats['total_found'].values())
        total_upscaled = sum(self.stats['upscaled'].values())
        total_skipped = sum(self.stats['skipped'].values())
        
        logger.info(f"\nTotal images found: {total_found}")
        logger.info(f"Total images upscaled: {total_upscaled}")
        logger.info(f"Total images skipped: {total_skipped}")
        
        logger.info("\nPer-category breakdown:")
        for category in self._bucket_names:
            found = self.stats['total_found'][category]
            upscaled = self.stats['upscaled'][category]
            skipped = self.stats['skipped'][category]
            errors = len(self.stats['errors'][category])
            
            if found > 0:
                logger.info(f"  {category}:")
                logger.info(f"    Found: {found}")
                logger.info(f"    Upscaled: {upscaled}")
                logger.info(f"    Skipped: {skipped}")
                if errors > 0:
                    logger.info(f"    Errors: {errors}")
        
        if any(self.stats['errors'].values()):
            logger.info("\nERRORS ENCOUNTERED:")
            for category, errors in self.stats['errors'].items():
                if errors:
                    logger.info(f"  {category}: {len(errors)} errors")
                    for error in errors[:5]:  # Show first 5 errors
                        logger.info(f"    - {error}")
                    if len(errors) > 5:
                        logger.info(f"    ... and {len(errors) - 5} more")
        
        logger.info("=" * 60)
    
    def run(self, cancel_event: Optional[threading.Event] = None) -> bool:
        """Main execution method. *cancel_event* is checked between categories and during file loops."""
        self._cancel_event = cancel_event
        try:
            return self._run_impl()
        finally:
            self._cancel_event = None

    def _run_impl(self) -> bool:
        log_startup_info(logger, "Small image upscaling process")
        logger.info(f"Review directory: {self.review_dir}")
        logger.info(f"Upscaled directory: {self.upscaled_dir}")
        logger.info(f"Target minimum dimension: {MIN_DIMENSION_TARGET}px")
        
        check_cancel_event(self._cancel_event)
        
        # Validate configuration
        if not self.validate_configuration():
            return False

        layout = layout_dict_for_discovery()
        self._bucket_names = discover_review_bucket_names(
            self.review_dir,
            explicit=layout["explicit"],
            class_qualifying_subdir=layout["class_qualifying_subdir"],
        )

        if not self._bucket_names:
            logger.warning(
                "No class subdirectories found under review dir (check data.class_names or layout under %s)",
                self.review_dir,
            )

        # Process each discovered bucket (same naming rules as raw_data class folders)
        for category in self._bucket_names:
            check_cancel_event(self._cancel_event)
            self.process_category(category)
        
        # Print summary
        self.print_summary()
        
        # Log completion
        total_upscaled = sum(self.stats['upscaled'].values())
        success = total_upscaled > 0 or sum(self.stats['total_found'].values()) == 0
        message = f"Upscaled {total_upscaled} images"
        log_completion_info(logger, success, message)
        
        return success


if __name__ == "__main__":
    from mb.cli import run_data_subcommand_cli

    sys.exit(run_data_subcommand_cli("upscale"))

