#!/usr/bin/env python3
"""
Deduplicate raw training files within and across class directories.

:class:`ImageDeduplicator` uses perceptual hashing, moves suspected duplicates to a
review area, and tracks cache state for incremental runs.

**CLI:** ``mb data deduplicate``; ``python -m mb.data.deduplicate`` delegates via
:func:`mb.cli.run_data_subcommand_cli`.
"""

import sys
import hashlib
import shutil
import io
import json
import pickle
import threading
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
# Image processing imports
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise ImportError("Error: PIL/Pillow not available. Image deduplication is required.")

# Import centralized logging configuration
from mb.utils.logging_setup import log_completion_info, log_startup_info, setup_logging
from mb.cancellation import check_cancel_event
from mb.data.class_layout import (
    CONVERTED_MEDIA_SUBDIR,
    discover_raw_data_bucket_names,
    layout_dict_for_discovery,
)
from mb.data.file_types import configured_media_suffixes
from mb.utils.snapshot import find_unified_snapshot, save_unified_snapshot

# Configure logging
logger = setup_logging(script_name="deduplicate_images")

# Default raw data directory
DEFAULT_RAW_DATA_DIR = Path("raw_data")


class ImageDeduplicator:
    """Handles deduplication of images across directories."""
    
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        
        # Statistics tracking
        self.stats = {
            'duplicates_removed': 0,
            'duplicates_found': 0,
            'small_images_removed': 0,
            'small_images_moved': 0,
        }
        
        # Review directory for small images (80-250px)
        self.review_dir = self.raw_data_dir / "small_images_review"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for storing file hashes (shared with :mod:`mb.data.gather`)
        self.cache_file = self.raw_data_dir / ".gather_cache.pkl"
        self.file_cache: Dict[str, str] = {}  # {file_path: hash_string} - optimized direct storage
        self.cache_modified = False
        self.duplicate_groups: List[Dict[str, object]] = []
        self.load_cache()
    
    def load_cache(self) -> None:
        """
        Load the cache from disk if it exists.
        Handles both old format (nested dict) and new format (direct hash) for backward compatibility.
        """
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                
                # Migrate old format to new format if needed
                if loaded_cache:
                    first_value = next(iter(loaded_cache.values()))
                    if isinstance(first_value, dict):
                        # Old format: migrate to new format
                        logger.info("Migrating cache from old format to new format...")
                        new_cache = {}
                        for file_path, cache_entry in loaded_cache.items():
                            if isinstance(cache_entry, dict):
                                hash_value = cache_entry.get('hash')
                                if hash_value:
                                    new_cache[file_path] = hash_value
                        self.file_cache = new_cache
                        logger.info(f"Migrated {len(new_cache)} entries to new format")
                        # Save migrated cache
                        self.cache_modified = True
                        self.save_cache()
                    else:
                        # New format: use directly
                        self.file_cache = loaded_cache
                
                logger.info(f"Loaded cache with {len(self.file_cache)} entries from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")
                self.file_cache = {}
        else:
            logger.info("No existing cache found. Starting fresh.")
            self.file_cache = {}
    
    def save_cache(self) -> None:
        """Save the cache to disk."""
        if not self.cache_modified:
            return
        
        try:
            # Create parent directory if it doesn't exist
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write cache to temporary file first, then rename (atomic operation)
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(self.file_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_file.replace(self.cache_file)
            self.cache_modified = False
            logger.debug(f"Saved cache with {len(self.file_cache)} entries to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_cached_hash(self, file_path: Path) -> Optional[str]:
        """
        Get cached hash for a file if it exists in cache.
        Optimized for direct hash storage (no nested dict access).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cached hash if available, None otherwise
        """
        return self.file_cache.get(str(file_path))
    
    def cache_hash(self, file_path: Path, file_hash: str) -> None:
        """
        Cache a file's hash.
        Optimized to store hash directly (no nested dict, no mtime tracking).
        
        Args:
            file_path: Path to the file
            file_hash: Hash value to cache
        """
        try:
            self.file_cache[str(file_path)] = file_hash
            self.cache_modified = True
        except Exception as e:
            logger.debug(f"Failed to cache hash for {file_path}: {e}")
    
    def calculate_file_hash(self, image_path: Path) -> Optional[str]:
        """
        Calculate a hash for an image based on its file content.
        Uses simple MD5 of file content - much faster than PIL-based hashing.
        Uses cache to avoid recalculating hashes for files that haven't changed.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MD5 hash string of the file content, or None if error
        """
        # Try to get from cache first
        cached_hash = self.get_cached_hash(image_path)
        if cached_hash:
            return cached_hash
        
        # Calculate hash if not in cache
        try:
            md5_hash = hashlib.md5()
            with open(image_path, 'rb') as f:
                # Read in chunks for memory efficiency with large files
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
            hash_value = md5_hash.hexdigest()
            
            # Cache the hash
            self.cache_hash(image_path, hash_value)
            return hash_value
        except Exception as e:
            logger.error(f"Error calculating file hash for {image_path}: {e}")
            return None
    
    def calculate_image_hash_pil(self, image_path: Path) -> str:
        """
        Calculate a hash for an image using PIL (more robust for different formats).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MD5 hash string of the image content
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB and resize to a standard size for consistent hashing
                img = img.convert('RGB')
                img = img.resize((64, 64))  # Small size for faster hashing
                
                # Convert to bytes and hash
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                return hashlib.md5(buffer.getvalue()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating PIL hash for {image_path}: {e}")
            return None
    
    def find_duplicates_in_directory(self, directory: Path) -> Dict[str, List[Path]]:
        """
        Find duplicate images within a single directory.
        
        Args:
            directory: Directory to scan for duplicates
            
        Returns:
            Dictionary mapping image hashes to lists of file paths
        """
        hash_to_files = defaultdict(list)
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return hash_to_files
        
        logger.info(f"Scanning for duplicates in: {directory}")
        
        # Use extension-specific globs to avoid expensive .is_file() checks
        for ext in configured_media_suffixes():
            for file_path in directory.rglob(f'*{ext}'):
                # No .is_file() check needed - rglob with pattern only returns files
                image_hash = self.calculate_file_hash(file_path)
                if image_hash:
                    hash_to_files[image_hash].append(file_path)
        
        # Filter to only include hashes with multiple files (duplicates)
        duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}
        
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate groups in {directory}")
            for hash_val, files in duplicates.items():
                logger.debug(f"Duplicate group ({len(files)} files): {[f.name for f in files]}")
        
        return duplicates
    
    def find_duplicates_across_directories(self, directories: List[Path]) -> Dict[str, List[Path]]:
        """
        Find duplicate images across multiple directories.
        
        Args:
            directories: List of directories to scan
            
        Returns:
            Dictionary mapping image hashes to lists of file paths from all directories
        """
        hash_to_files = defaultdict(list)
        
        for directory in directories:
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
                
            logger.info(f"Scanning for duplicates in: {directory}")
            
            # Use extension-specific globs to avoid expensive .is_file() checks
            for ext in configured_media_suffixes():
                for file_path in directory.rglob(f'*{ext}'):
                    # No .is_file() check needed - rglob with pattern only returns files
                    image_hash = self.calculate_file_hash(file_path)
                    if image_hash:
                        hash_to_files[image_hash].append(file_path)
        
        # Filter to only include hashes with multiple files (duplicates)
        duplicates = {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}
        
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate groups across all directories")
        
        return duplicates
    
    def remove_duplicates_from_directory(self, directory: Path, keep_first: bool = True) -> int:
        """
        Remove duplicate images from a directory, keeping the first occurrence.
        
        Args:
            directory: Directory to clean
            keep_first: If True, keep the first file in each duplicate group
            
        Returns:
            Number of files removed
        """
        duplicates = self.find_duplicates_in_directory(directory)
        removed_count = 0
        
        for hash_val, files in duplicates.items():
            if keep_first:
                # Keep the first file, remove the rest
                files_to_remove = files[1:]
            else:
                # Keep the last file, remove the rest
                files_to_remove = files[:-1]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed duplicate: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing duplicate {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate files from {directory}")
        
        return removed_count
    
    def process_small_images_from_directory(self, directory: Path) -> Tuple[int, int]:
        """
        Process images from a directory:
        - Remove images with any dimension < 80px
        - Move images with dimensions between 80px and 250px to review directory
        
        Args:
            directory: Directory to clean
            
        Returns:
            Tuple of (removed_count, moved_count)
        """
        removed_count = 0
        moved_count = 0
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return removed_count, moved_count
        
        logger.info(f"Scanning {directory} for small images...")
        
        # Use extension-specific globs to avoid expensive .is_file() checks
        for ext in configured_media_suffixes():
            for file_path in directory.rglob(f'*{ext}'):
                # No .is_file() check needed - rglob with pattern only returns files
                try:
                    # Open image and get dimensions, ensuring proper cleanup on Windows
                    img = None
                    width = height = 0
                    try:
                        img = Image.open(file_path)
                        # Force PIL to read image data to avoid Windows file locking issues
                        img.load()
                        width, height = img.size
                    finally:
                        # Explicitly close the image to release file handle on Windows
                        if img:
                            img.close()
                    
                    min_dim = min(width, height)
                    
                    # Remove if minimum dimension < 80px
                    if min_dim < 80:
                        # Small delay to ensure file handle is released on Windows
                        time.sleep(0.01)
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed very small image: {file_path} ({width}x{height})")
                    # Move if minimum dimension is between 80px and 250px
                    elif min_dim < 250:
                        # Create target path preserving relative structure
                        relative_path = file_path.relative_to(directory)
                        # Use directory name to preserve source context
                        source_dir_name = directory.name
                        target_path = self.review_dir / source_dir_name / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Handle filename conflicts
                        counter = 1
                        original_target = target_path
                        while target_path.exists():
                            stem = original_target.stem
                            suffix = original_target.suffix
                            target_path = original_target.parent / f"{stem}_{counter}{suffix}"
                            counter += 1
                        
                        # Small delay to ensure file handle is released on Windows
                        time.sleep(0.01)
                        shutil.move(str(file_path), str(target_path))
                        moved_count += 1
                        logger.debug(f"Moved small image for review: {file_path} ({width}x{height}) -> {target_path}")
                except Exception as e:
                    logger.error(f"Error checking dimensions for {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} very small images (< 80px) from {directory}")
        if moved_count > 0:
            logger.info(f"Moved {moved_count} small images (80-250px) from {directory} to review directory")
        
        return removed_count, moved_count
    
    def _discover_converted_directories(self) -> List[Path]:
        """Return class-scoped ``CONVERTED`` directories eligible for deduplication."""
        layout = layout_dict_for_discovery()
        bucket_names = discover_raw_data_bucket_names(
            self.raw_data_dir,
            explicit=layout["explicit"],
            class_qualifying_subdir=layout["class_qualifying_subdir"],
        )
        converted_directories: List[Path] = []
        skipped_without_converted: List[Path] = []
        for bucket_name in bucket_names:
            class_dir = self.raw_data_dir / bucket_name
            converted_dir = class_dir / CONVERTED_MEDIA_SUBDIR
            if converted_dir.is_dir():
                converted_directories.append(converted_dir)
            else:
                skipped_without_converted.append(class_dir)

        if skipped_without_converted:
            logger.info(
                "Skipping %d class directories without %s",
                len(skipped_without_converted),
                CONVERTED_MEDIA_SUBDIR,
            )
            for skipped in skipped_without_converted:
                logger.debug("Skipped (missing CONVERTED): %s", skipped)
        return converted_directories

    def _serialize_duplicate_groups(self, duplicates: Dict[str, List[Path]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for hash_value in sorted(duplicates.keys()):
            files = duplicates[hash_value]
            out.append(
                {
                    "hash": hash_value,
                    "files": sorted(str(p) for p in files),
                }
            )
        return out

    def _update_snapshot_with_duplicates(self, *, run_id: Optional[str]) -> None:
        """Persist deduplication metadata into unified snapshot when available."""
        snapshot = find_unified_snapshot([self.raw_data_dir], run_id=run_id, logger=logger)
        if not snapshot:
            logger.warning("No unified snapshot found under %s; skipping deduplication snapshot update", self.raw_data_dir)
            return
        snapshot.set_deduplication_results(self.duplicate_groups)
        saved = save_unified_snapshot(snapshot, self.raw_data_dir, logger=logger)
        if saved is not None:
            logger.info("Updated snapshot with deduplication results: %s", saved)

    def perform_deduplication(self, *, list_only: bool = False) -> None:
        """Perform deduplication only within class ``CONVERTED`` directories under raw data."""
        check_cancel_event(getattr(self, "_cancel_event", None))
        logger.info("=" * 60)
        logger.info("STARTING DEDUPLICATION PROCESS")
        logger.info("=" * 60)
        converted_directories = self._discover_converted_directories()

        if not converted_directories:
            logger.warning("No CONVERTED directories found for deduplication")
            self.duplicate_groups = []
            return

        logger.info("Processing CONVERTED directories: %s", [str(d) for d in converted_directories])
        if list_only:
            logger.info(
                "\nList-only mode: still removes tiny images and duplicates *within* each "
                "CONVERTED folder; only *cross-class* duplicate groups are kept for review."
            )
            logger.info("\nStep 0: Processing small images...")
            logger.info("  - Removing images with any dimension < 80px")
            logger.info("  - Moving images with dimensions between 80px and 250px to review directory")
            for directory in converted_directories:
                check_cancel_event(getattr(self, "_cancel_event", None))
                removed_count, moved_count = self.process_small_images_from_directory(directory)
                self.stats["small_images_removed"] += removed_count
                self.stats["small_images_moved"] += moved_count

            logger.info("\nStep 1: Removing duplicates within each CONVERTED directory...")
            for directory in converted_directories:
                check_cancel_event(getattr(self, "_cancel_event", None))
                removed_count = self.remove_duplicates_from_directory(directory)
                self.stats["duplicates_removed"] += removed_count

            logger.info("\nStep 2: Finding duplicates across CONVERTED directories (review-only)...")
            check_cancel_event(getattr(self, "_cancel_event", None))
            all_duplicates = self.find_duplicates_across_directories(converted_directories)
            self.stats["duplicates_found"] = len(all_duplicates)
            self.duplicate_groups = self._serialize_duplicate_groups(all_duplicates)
            logger.info("=" * 60)
            logger.info("DEDUPLICATION LIST-ONLY SCAN COMPLETED")
            logger.info("=" * 60)
            return

        # Step 0: Process small images
        # - Remove images with any dimension < 80px
        # - Move images with dimensions between 80px and 250px to review directory
        logger.info("\nStep 0: Processing small images...")
        logger.info("  - Removing images with any dimension < 80px")
        logger.info("  - Moving images with dimensions between 80px and 250px to review directory")
        for directory in converted_directories:
            check_cancel_event(getattr(self, "_cancel_event", None))
            removed_count, moved_count = self.process_small_images_from_directory(directory)
            self.stats['small_images_removed'] += removed_count
            self.stats['small_images_moved'] += moved_count
        
        # Step 1: Remove duplicates within each directory
        logger.info("\nStep 1: Removing duplicates within each directory...")
        for directory in converted_directories:
            check_cancel_event(getattr(self, "_cancel_event", None))
            removed_count = self.remove_duplicates_from_directory(directory)
            self.stats['duplicates_removed'] += removed_count
        
        # Step 2: Find duplicates across all directories
        logger.info("\nStep 2: Finding duplicates across all directories...")
        check_cancel_event(getattr(self, "_cancel_event", None))
        all_duplicates = self.find_duplicates_across_directories(converted_directories)
        self.stats['duplicates_found'] = len(all_duplicates)
        self.duplicate_groups = self._serialize_duplicate_groups(all_duplicates)
        
        if all_duplicates:
            logger.info(f"Found {len(all_duplicates)} duplicate groups across all directories")
            for hash_val, files in all_duplicates.items():
                logger.info(f"Duplicate group ({len(files)} files):")
                for file_path in files:
                    logger.info(f"  - {file_path}")

        logger.info("=" * 60)
        logger.info("DEDUPLICATION PROCESS COMPLETED")
        logger.info("=" * 60)
    
    def print_summary(self) -> None:
        """Print a summary of the deduplication operation."""
        logger.info("=" * 60)
        logger.info("DEDUPLICATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Very small images removed (< 80px): {self.stats['small_images_removed']}")
        logger.info(f"Small images moved to review (80-250px): {self.stats['small_images_moved']}")
        if self.stats['small_images_moved'] > 0:
            logger.info(f"Review directory: {self.review_dir}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"Duplicate groups found across directories: {self.stats['duplicates_found']}")
        logger.info("=" * 60)
    
    def run(
        self,
        cancel_event: Optional[threading.Event] = None,
        *,
        list_only: bool = False,
        run_id: Optional[str] = None,
    ) -> bool:
        """Main execution method. *cancel_event* is checked between major steps."""
        self._cancel_event = cancel_event
        try:
            return self._run_impl(list_only=list_only, run_id=run_id)
        finally:
            self._cancel_event = None

    def _run_impl(self, *, list_only: bool = False, run_id: Optional[str] = None) -> bool:
        log_startup_info(logger, "Image deduplication process")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        if list_only:
            logger.info(
                "Mode: list-only (cross-class duplicates for review; within-CONVERTED duplicates removed)"
            )
        
        check_cancel_event(self._cancel_event)
        
        # Validate raw data directory
        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory does not exist: {self.raw_data_dir}")
            return False
        
        # Perform deduplication
        self.perform_deduplication(list_only=list_only)

        # Update snapshot duplicate metadata when a snapshot is available.
        self._update_snapshot_with_duplicates(run_id=run_id)
        
        # Save cache before finishing
        if self.cache_modified:
            self.save_cache()
            logger.info("Cache saved after deduplication")
        
        # Print summary
        self.print_summary()
        
        # Log completion
        success = True
        if list_only:
            message = (
                f"List-only dedup: removed {self.stats['duplicates_removed']} within-class duplicates, "
                f"found {self.stats['duplicates_found']} cross-class duplicate groups for review"
            )
        else:
            message = f"Removed {self.stats['small_images_removed']} very small images, moved {self.stats['small_images_moved']} to review, removed {self.stats['duplicates_removed']} duplicate files, found {self.stats['duplicates_found']} duplicate groups"
        log_completion_info(logger, success, message)
        
        return success

    def duplicate_groups_as_json(self) -> str:
        """Return duplicate groups as pretty JSON for CLI/UI consumption."""
        return json.dumps(self.duplicate_groups, indent=2)


if __name__ == "__main__":
    from mb.cli import run_data_subcommand_cli

    sys.exit(run_data_subcommand_cli("deduplicate"))

