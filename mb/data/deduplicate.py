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

# Configure logging
logger = setup_logging(script_name="deduplicate_images")

# Image file extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

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
            'coherent_duplicates_found': 0,
            'small_images_removed': 0,
            'small_images_moved': 0,
        }
        
        # Review directory for small images (80-250px)
        self.review_dir = self.raw_data_dir / "small_images_review"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for storing file hashes (shared with gather_coherent_images.py)
        self.cache_file = self.raw_data_dir / ".gather_cache.pkl"
        self.file_cache: Dict[str, str] = {}  # {file_path: hash_string} - optimized direct storage
        self.cache_modified = False
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
        for ext in IMAGE_EXTENSIONS:
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
            for ext in IMAGE_EXTENSIONS:
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
    
    def check_coherent_against_other_directories(self, coherent_dir: Path, other_dirs: List[Path]) -> List[Path]:
        """
        Check if any images in the coherent directory are duplicates of images in other directories.
        
        Args:
            coherent_dir: Directory containing coherent images
            other_dirs: List of directories to check against (incoherent, semi-incoherent)
            
        Returns:
            List of coherent image paths that are duplicates of images in other directories
        """
        logger.info("Checking coherent images against other directories for duplicates...")
        
        # Get hashes of all images in other directories
        other_hashes = set()
        for directory in other_dirs:
            if not directory.exists():
                continue
                
            # Use extension-specific globs to avoid expensive .is_file() checks
            for ext in IMAGE_EXTENSIONS:
                for file_path in directory.rglob(f'*{ext}'):
                    # No .is_file() check needed - rglob with pattern only returns files
                    image_hash = self.calculate_file_hash(file_path)
                    if image_hash:
                        other_hashes.add(image_hash)
        
        logger.info(f"Found {len(other_hashes)} unique images in other directories")
        
        # Check coherent images against other hashes
        duplicate_coherent_files = []
        coherent_files_checked = 0
        
        # Use extension-specific globs to avoid expensive .is_file() checks
        for ext in IMAGE_EXTENSIONS:
            for file_path in coherent_dir.rglob(f'*{ext}'):
                # No .is_file() check needed - rglob with pattern only returns files
                coherent_files_checked += 1
                image_hash = self.calculate_file_hash(file_path)
                if image_hash and image_hash in other_hashes:
                    duplicate_coherent_files.append(file_path)
                    logger.warning(f"Coherent image {file_path.name} is duplicate of image in other directories")
        
        logger.info(f"Checked {coherent_files_checked} coherent images")
        logger.info(f"Found {len(duplicate_coherent_files)} coherent images that are duplicates of images in other directories")
        
        return duplicate_coherent_files
    
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
        for ext in IMAGE_EXTENSIONS:
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
    
    def perform_deduplication(self) -> None:
        """Perform comprehensive deduplication across all directories."""
        check_cancel_event(getattr(self, "_cancel_event", None))
        logger.info("=" * 60)
        logger.info("STARTING DEDUPLICATION PROCESS")
        logger.info("=" * 60)
        
        # Define directory paths
        coherent_dir = self.raw_data_dir / "coherent"
        incoherent_dir = self.raw_data_dir / "incoherent"
        semi_incoherent_dir = self.raw_data_dir / "semi-incoherent"
        
        # Explicitly exclude rejected directory and review directories
        rejected_dir = self.raw_data_dir / "rejected"
        review_dir = self.raw_data_dir / "small_images_review"
        
        directories_to_check = [coherent_dir, incoherent_dir, semi_incoherent_dir]
        # Filter out any directories that shouldn't be processed
        excluded_dirs = {rejected_dir, review_dir}
        existing_directories = [d for d in directories_to_check if d.exists() and d not in excluded_dirs]
        
        # Log excluded directories for clarity
        if rejected_dir.exists():
            logger.info(f"Excluding rejected directory from processing: {rejected_dir}")
        if review_dir.exists():
            logger.info(f"Excluding review directory from processing: {review_dir}")
        
        if not existing_directories:
            logger.warning("No directories found for deduplication")
            return
        
        logger.info(f"Processing directories: {[d.name for d in existing_directories]}")
        
        # Step 0: Process small images
        # - Remove images with any dimension < 80px
        # - Move images with dimensions between 80px and 250px to review directory
        logger.info("\nStep 0: Processing small images...")
        logger.info("  - Removing images with any dimension < 80px")
        logger.info("  - Moving images with dimensions between 80px and 250px to review directory")
        for directory in existing_directories:
            check_cancel_event(getattr(self, "_cancel_event", None))
            removed_count, moved_count = self.process_small_images_from_directory(directory)
            self.stats['small_images_removed'] += removed_count
            self.stats['small_images_moved'] += moved_count
        
        # Step 1: Remove duplicates within each directory
        logger.info("\nStep 1: Removing duplicates within each directory...")
        for directory in existing_directories:
            check_cancel_event(getattr(self, "_cancel_event", None))
            removed_count = self.remove_duplicates_from_directory(directory)
            self.stats['duplicates_removed'] += removed_count
        
        # Step 2: Find duplicates across all directories
        logger.info("\nStep 2: Finding duplicates across all directories...")
        check_cancel_event(getattr(self, "_cancel_event", None))
        all_duplicates = self.find_duplicates_across_directories(existing_directories)
        self.stats['duplicates_found'] = len(all_duplicates)
        
        if all_duplicates:
            logger.info(f"Found {len(all_duplicates)} duplicate groups across all directories")
            for hash_val, files in all_duplicates.items():
                logger.info(f"Duplicate group ({len(files)} files):")
                for file_path in files:
                    logger.info(f"  - {file_path}")
        
        # Step 3: Check coherent images against other directories
        logger.info("\nStep 3: Checking coherent images against other directories...")
        check_cancel_event(getattr(self, "_cancel_event", None))
        if coherent_dir.exists() and (incoherent_dir.exists() or semi_incoherent_dir.exists()):
            other_dirs = [d for d in [incoherent_dir, semi_incoherent_dir] if d.exists()]
            duplicate_coherent_files = self.check_coherent_against_other_directories(coherent_dir, other_dirs)
            
            if duplicate_coherent_files:
                logger.warning(f"Found {len(duplicate_coherent_files)} coherent images that are duplicates of images in other directories")
                logger.warning("These should be reviewed and potentially removed from the coherent dataset")
                for file_path in duplicate_coherent_files:
                    logger.warning(f"  - {file_path}")
                self.stats['coherent_duplicates_found'] = len(duplicate_coherent_files)
            else:
                logger.info("No coherent images found that are duplicates of images in other directories")
        
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
        logger.info(f"Coherent images that are duplicates of other directories: {self.stats['coherent_duplicates_found']}")
        logger.info("=" * 60)
    
    def run(self, cancel_event: Optional[threading.Event] = None) -> bool:
        """Main execution method. *cancel_event* is checked between major steps."""
        self._cancel_event = cancel_event
        try:
            return self._run_impl()
        finally:
            self._cancel_event = None

    def _run_impl(self) -> bool:
        log_startup_info(logger, "Image deduplication process")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        
        check_cancel_event(self._cancel_event)
        
        # Validate raw data directory
        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory does not exist: {self.raw_data_dir}")
            return False
        
        # Perform deduplication
        self.perform_deduplication()
        
        # Save cache before finishing
        if self.cache_modified:
            self.save_cache()
            logger.info("Cache saved after deduplication")
        
        # Print summary
        self.print_summary()
        
        # Log completion
        success = True
        message = f"Removed {self.stats['small_images_removed']} very small images, moved {self.stats['small_images_moved']} to review, removed {self.stats['duplicates_removed']} duplicate files, found {self.stats['duplicates_found']} duplicate groups"
        log_completion_info(logger, success, message)
        
        return success


if __name__ == "__main__":
    from mb.cli import run_data_subcommand_cli

    sys.exit(run_data_subcommand_cli("deduplicate"))

