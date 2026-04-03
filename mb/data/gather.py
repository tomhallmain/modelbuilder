#!/usr/bin/env python3
"""
Gather images into the raw data tree for training.

This is the **first** step in the usual image pipeline: **gather** → **convert**
→ **create-dataset** (see :func:`mb.pipeline_config.gather_pipeline_defaults` and
``data.gather`` in pipeline YAML).

Randomly selects images from specified subdirectories up to a target limit,
and copies them into the configured target directory (see ``data.gather.default_target_dir``)
preserving original format.

**CLI:** ``mb data gather``; ``python -m mb.data.gather`` delegates to the same parser
via :func:`mb.cli.run_data_subcommand_cli`.

Features:
- Supports subsequent runs: filters out already-processed images (in target and rejected directories)
- Target count is treated as a limit, not an exact requirement
- Each run creates a unique timestamped subdirectory for new files
- Tracks total count across all subdirectories
- Supports rejected directory for manually rejected images
- Preserves original file extensions and content (no format conversion)
"""

import sys
import random
import hashlib
import shutil
import pickle
import os
import threading
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Image processing imports
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise ImportError("Error: PIL/Pillow not available. Image conversion is required, but currently not available.")

# Import centralized logging configuration
from mb.utils.logging_setup import log_completion_info, log_startup_info, setup_logging
from mb.cancellation import check_cancel_event
from mb.data.class_layout import (
    discover_raw_data_bucket_names,
    layout_dict_for_discovery,
    normalize_qualifying_subdir,
)

# Configure logging
logger = setup_logging(script_name="gather_images")

# Image file extensions for source scans (pre-convert; not identical to pipeline data.image_types).
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
TARGET_EXTENSIONS = {'.jpg', '.jpeg'}

class ImageGatherer:
    """Gathers images into a run-scoped target tree under the configured staging directory."""
    
    def __init__(
        self,
        source_dir: str,
        valid_subdirs: List[str],
        target_dir: Path,
        target_count: int,
        rejected_dir: Path = None,
        subdir_weights: Dict[str, float] = None,
        class_qualifying_subdir: Optional[str] = None,
    ):
        self.source_dir = Path(source_dir)
        self.valid_subdirs = set(valid_subdirs)
        self.class_qualifying_subdir = class_qualifying_subdir
        self.target_dir = target_dir
        self.rejected_dir = Path(rejected_dir) if rejected_dir else None
        self.subdir_weights = subdir_weights or {}
        
        # Validate target count
        if target_count <= 0:
            raise ValueError("Target count must be positive")
        self.target_count = target_count
        
        # Create unique run identifier based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique subdirectories for this run (target only, rejected is flat)
        self.run_target_dir = self.target_dir / self.run_id
        
        # Statistics tracking
        self.stats = {
            'total_files_found': 0,
            'files_processed': 0,
            'files_copied': 0,
            'files_converted': 0,
            'errors': defaultdict(list),
            'skipped_files': [],
            'duplicates_removed': 0,
            'duplicates_found': 0,
            'already_processed_count': 0,
            'current_total_in_target': 0,
            'files_needed': 0
        }
        
        # Ensure directories exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.run_target_dir.mkdir(parents=True, exist_ok=True)
        # Rejected directory is flat (no run-specific subdirectories)
        if self.rejected_dir:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for storing file hashes
        self.cache_file = self.target_dir.parent / ".gather_cache.pkl"
        self.file_cache: Dict[str, str] = {}  # {file_path: hash_string} - optimized direct storage
        self.cache_modified = False
        
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
    
    def validate_configuration(self) -> bool:
        """Validate the configuration and source directory."""
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return False
            
        if not self.source_dir.is_dir():
            logger.error(f"Source path is not a directory: {self.source_dir}")
            return False
            
        if not self.valid_subdirs:
            logger.error("No valid subdirectories specified")
            return False
            
        # Check if valid subdirectories exist
        missing_subdirs = []
        for subdir in self.valid_subdirs:
            subdir_path = self.source_dir / subdir
            logger.info(f"Checking subdirectory: {subdir_path}")
            if not subdir_path.exists():
                missing_subdirs.append(subdir)
                
        if missing_subdirs:
            logger.error(f"Specified subdirectories do not exist: {missing_subdirs}")
            return False

        q = normalize_qualifying_subdir(self.class_qualifying_subdir)
        if q:
            bad_qual: List[str] = []
            for subdir in self.valid_subdirs:
                qual_path = self.source_dir / subdir / q
                if not qual_path.is_dir():
                    bad_qual.append(str(qual_path))
            if bad_qual:
                logger.error(
                    "Required qualifying subdirectory %r under each of --subdirs; missing or not a directory: %s",
                    q,
                    bad_qual,
                )
                return False

        # Check if target directory is writable
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.target_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            logger.error(f"Cannot write to target directory {self.target_dir}: {e}")
            return False
            
        return True
    
    def get_already_processed_hashes(self) -> Set[str]:
        """
        Get hashes of all images already processed (in target directory and rejected folders).
        Uses cache to avoid recalculating hashes for unchanged files.
        
        Returns:
            Set of image hashes that have already been processed
        """
        already_processed = set()
        target_count = 0
        rejected_count = 0
        cache_hits = 0
        cache_misses = 0
        
        # Check target directory (all subdirectories)
        if self.target_dir.exists():
            logger.info("Scanning target directory for already-processed images...")
            # Use extension-specific globs to avoid expensive .is_file() checks
            for ext in IMAGE_EXTENSIONS:
                for file_path in self.target_dir.rglob(f'*{ext}'):
                    # No .is_file() check needed - rglob with pattern only returns files
                    # Try to get from cache first
                    image_hash = self.get_cached_hash(file_path)
                    if image_hash:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                        image_hash = self.calculate_image_hash(file_path)
                        if image_hash:
                            self.cache_hash(file_path, image_hash)
                    
                    if image_hash:
                        already_processed.add(image_hash)
                        target_count += 1
            logger.info(f"Found {target_count} already-processed images in target directory (cache: {cache_hits} hits, {cache_misses} misses)")
        
        # Check rejected directory (flat structure, no subdirectories)
        if self.rejected_dir and self.rejected_dir.exists():
            logger.info("Scanning rejected directory for already-processed images...")
            # Only check files directly in rejected directory, not subdirectories
            # Check extension first (cheap) before .is_file() (expensive)
            for file_path in self.rejected_dir.iterdir():
                if file_path.suffix.lower() in IMAGE_EXTENSIONS and file_path.is_file():
                    # Try to get from cache first
                    image_hash = self.get_cached_hash(file_path)
                    if image_hash:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                        image_hash = self.calculate_image_hash(file_path)
                        if image_hash:
                            self.cache_hash(file_path, image_hash)
                    
                    if image_hash:
                        already_processed.add(image_hash)
                        rejected_count += 1
            logger.info(f"Found {rejected_count} already-processed images in rejected directory")
        
        # Save cache after scanning processed directories
        if self.cache_modified:
            self.save_cache()
        
        self.stats['already_processed_count'] = len(already_processed)
        logger.info(f"Total unique already-processed images: {len(already_processed)}")
        return already_processed
    
    def count_existing_files_in_target(self) -> int:
        """Count total files in target directory (across all subdirectories)."""
        count = 0
        if self.target_dir.exists():
            # Use extension-specific globs to avoid expensive .is_file() checks
            for ext in IMAGE_EXTENSIONS:
                count += len(list(self.target_dir.rglob(f'*{ext}')))
        return count
    
    def find_image_files(self) -> Dict[str, List[Path]]:
        """
        Find all image files in valid subdirectories, excluding already-processed ones.
        Returns a dictionary mapping subdirectory names to lists of file paths.
        """
        files_by_subdir = defaultdict(list)
        
        # Get hashes of already-processed files (convert to frozenset for slightly faster lookups)
        already_processed_hashes = frozenset(self.get_already_processed_hashes())
        
        # Count existing files in target
        self.stats['current_total_in_target'] = self.count_existing_files_in_target()
        self.stats['files_needed'] = max(0, self.target_count - self.stats['current_total_in_target'])
        
        logger.info(f"Current total in target directory: {self.stats['current_total_in_target']}")
        logger.info(f"Target limit: {self.target_count}")
        logger.info(f"Files needed to reach target: {self.stats['files_needed']}")
        
        if self.stats['files_needed'] == 0:
            logger.info("Target limit already reached. No new files needed.")
            return {}
        
        files_checked = 0
        cache_hits = 0
        cache_misses = 0
        unprocessed_found = 0
        
        for subdir in self.valid_subdirs:
            subdir_path = self.source_dir / subdir
            if not subdir_path.exists():
                logger.error(f"Subdirectory does not exist: {subdir_path}")
                continue
                
            logger.info(f"Scanning subdirectory: {subdir}")
            
            try:
                # Use extension-specific globs to avoid expensive .is_file() checks
                # rglob with patterns only matches files, not directories
                for ext in IMAGE_EXTENSIONS:
                    for file_path in subdir_path.rglob(f'*{ext}'):
                        # No .is_file() check needed - rglob with pattern only returns files
                        files_checked += 1
                        
                        # Log progress every 1000 images
                        if files_checked % 1000 == 0:
                            check_cancel_event(getattr(self, "_cancel_event", None))
                            logger.info(f"Progress: Checked {files_checked} images, found {unprocessed_found} unprocessed... (cache: {cache_hits} hits, {cache_misses} misses)")
                            # Save cache every 1000 files
                            if self.cache_modified:
                                self.save_cache()
                                logger.debug("Cache saved (every 1000 files)")
                        
                        # Try to get hash from cache first
                        image_hash = self.get_cached_hash(file_path)
                        if image_hash:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                            # Calculate hash if not in cache
                            image_hash = self.calculate_image_hash(file_path)
                            if image_hash:
                                self.cache_hash(file_path, image_hash)
                        
                        # Check if this file has already been processed
                        if image_hash and image_hash not in already_processed_hashes:
                            files_by_subdir[subdir].append(file_path)
                            unprocessed_found += 1
                        # Note: Removed debug logging for skipped files to reduce overhead
                        
            except PermissionError as e:
                logger.error(f"Permission denied accessing {subdir_path}: {e}")
                self.stats['errors']['permission'].append(str(subdir_path))
            except Exception as e:
                logger.error(f"Error scanning {subdir_path}: {e}")
                self.stats['errors']['scanning'].append((str(subdir_path), str(e)))
            
            # Save cache at the end of each subdirectory
            if self.cache_modified:
                self.save_cache()
                logger.debug(f"Cache saved (end of subdirectory: {subdir})")
        
        total_files = sum(len(files) for files in files_by_subdir.values())
        self.stats['total_files_found'] = total_files
        
        logger.info(f"Finished scanning. Checked {files_checked} total images (cache: {cache_hits} hits, {cache_misses} misses).")
        
        # Log files found per subdirectory
        for subdir, files in files_by_subdir.items():
            logger.info(f"Found {len(files)} files in subdirectory '{subdir}'")
        
        logger.info(f"Found {total_files} unprocessed image files across all subdirectories")
        
        # Check if any image files were found
        if total_files == 0:
            logger.warning("No unprocessed image files found in any of the specified subdirectories")
            return {}
            
        return files_by_subdir
    
    def select_random_files(self, files_by_subdir: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """
        Randomly select files up to the target count limit, respecting subdirectory weights.
        Target count is treated as a limit, not an exact requirement.
        
        Args:
            files_by_subdir: Dictionary mapping subdirectory names to lists of file paths
            
        Returns:
            Dictionary mapping subdirectory names to lists of selected file paths
        """
        if not files_by_subdir:
            logger.warning("No image files found to select from")
            return {}
        
        # Calculate how many files we need to reach the target limit
        files_needed = self.stats['files_needed']
        
        if files_needed <= 0:
            logger.info("Target limit already reached. No files to select.")
            return {}
        
        # Flatten all files to check total
        all_files = []
        for files in files_by_subdir.values():
            all_files.extend(files)
        
        if len(all_files) <= files_needed:
            logger.info(f"Found {len(all_files)} unprocessed files, selecting all of them (need {files_needed} to reach target)")
            return files_by_subdir  # Return all files grouped by subdirectory
        
        # If weights are specified, use weighted sampling
        if self.subdir_weights:
            return self._select_weighted_files(files_by_subdir, files_needed)
        else:
            # No weights specified, use uniform random sampling
            logger.info(f"Randomly selecting {files_needed} files from {len(all_files)} available to reach target limit")
            # Sample proportionally from each subdirectory
            selected_by_subdir = {}
            total_files = len(all_files)
            for subdir, files in files_by_subdir.items():
                # Calculate proportional sample size for this subdirectory
                subdir_proportion = len(files) / total_files
                subdir_target = max(1, int(round(files_needed * subdir_proportion)))
                subdir_target = min(subdir_target, len(files))
                
                if subdir_target > 0:
                    selected_by_subdir[subdir] = random.sample(files, subdir_target)
            
            # Adjust if we're short due to rounding
            total_selected = sum(len(files) for files in selected_by_subdir.values())
            if total_selected < files_needed:
                # Add remaining files randomly from all subdirectories
                remaining_files = []
                for subdir, files in files_by_subdir.items():
                    selected = set(selected_by_subdir.get(subdir, []))
                    remaining_files.extend([f for f in files if f not in selected])
                
                if remaining_files:
                    additional_needed = files_needed - total_selected
                    additional_count = min(additional_needed, len(remaining_files))
                    additional_selected = random.sample(remaining_files, additional_count)
                    
                    # Add to appropriate subdirectories
                    for file_path in additional_selected:
                        # Find which subdirectory this file belongs to
                        for subdir, files in files_by_subdir.items():
                            if file_path in files:
                                if subdir not in selected_by_subdir:
                                    selected_by_subdir[subdir] = []
                                selected_by_subdir[subdir].append(file_path)
                                break
            
            return selected_by_subdir
    
    def _select_weighted_files(self, files_by_subdir: Dict[str, List[Path]], files_needed: int) -> Dict[str, List[Path]]:
        """
        Select files proportionally based on subdirectory weights.
        
        Args:
            files_by_subdir: Dictionary mapping subdirectory names to lists of file paths
            files_needed: Number of files to select
            
        Returns:
            Dictionary mapping subdirectory names to lists of selected file paths
        """
        # Normalize weights to sum to 1.0 (any positive numbers work as relative weights)
        # Example: weights 4 and 3 become 4/7=57.14% and 3/7=42.86%
        total_weight = sum(self.subdir_weights.get(subdir, 1.0) for subdir in files_by_subdir.keys())
        normalized_weights = {subdir: self.subdir_weights.get(subdir, 1.0) / total_weight 
                             for subdir in files_by_subdir.keys()}
        
        logger.info("Using weighted sampling with normalized weights:")
        for subdir in files_by_subdir.keys():
            original_weight = self.subdir_weights.get(subdir, 1.0)
            normalized_weight = normalized_weights[subdir]
            logger.info(f"  {subdir}: weight {original_weight} -> {normalized_weight:.2%}")
        
        selected_by_subdir = {}
        
        # Calculate target counts per subdirectory based on weights
        subdir_targets = {}
        for subdir, files in files_by_subdir.items():
            weight = normalized_weights[subdir]
            target_count = int(round(files_needed * weight))
            # Ensure we don't exceed available files
            target_count = min(target_count, len(files))
            subdir_targets[subdir] = target_count
        
        # Adjust if rounding caused total to be off
        total_targeted = sum(subdir_targets.values())
        if total_targeted < files_needed:
            # Distribute remaining files to subdirectories with available files
            remaining = files_needed - total_targeted
            for subdir in sorted(files_by_subdir.keys(), key=lambda x: normalized_weights[x], reverse=True):
                if remaining <= 0:
                    break
                available = len(files_by_subdir[subdir]) - subdir_targets[subdir]
                if available > 0:
                    add_count = min(remaining, available)
                    subdir_targets[subdir] += add_count
                    remaining -= add_count
        
        # Sample from each subdirectory according to targets
        for subdir, files in files_by_subdir.items():
            target_count = subdir_targets[subdir]
            if target_count > 0:
                if len(files) <= target_count:
                    logger.info(f"Selecting all {len(files)} files from '{subdir}'")
                    selected_by_subdir[subdir] = files
                else:
                    logger.info(f"Selecting {target_count} files from {len(files)} available in '{subdir}'")
                    selected_by_subdir[subdir] = random.sample(files, target_count)
        
        # If we still need more files (due to rounding), randomly select from remaining
        total_selected = sum(len(files) for files in selected_by_subdir.values())
        if total_selected < files_needed:
            remaining_files = []
            for subdir, files in files_by_subdir.items():
                selected = set(selected_by_subdir.get(subdir, []))
                remaining_files.extend([f for f in files if f not in selected])
            
            if remaining_files:
                additional_needed = files_needed - total_selected
                additional_count = min(additional_needed, len(remaining_files))
                logger.info(f"Selecting {additional_count} additional files to reach target")
                additional_selected = random.sample(remaining_files, additional_count)
                
                # Add to appropriate subdirectories
                for file_path in additional_selected:
                    # Find which subdirectory this file belongs to
                    for subdir, files in files_by_subdir.items():
                        if file_path in files:
                            if subdir not in selected_by_subdir:
                                selected_by_subdir[subdir] = []
                            selected_by_subdir[subdir].append(file_path)
                            break
        
        total_selected = sum(len(files) for files in selected_by_subdir.values())
        logger.info(f"Selected {total_selected} files total using weighted sampling")
        
        return selected_by_subdir
    
    def convert_to_jpg(self, source_path: Path, target_path: Path) -> bool:
        """Convert image to JPG format."""
        try:
            # Check if source file exists and is readable
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
                
                # Save as JPG
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
            logger.error(f"OS error converting {source_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error converting {source_path}: {e}")
            return False
    
    def copy_file(self, source_path: Path, target_path: Path) -> bool:
        """Copy a file to the target directory, preserving original extension and content."""
        try:
            # Check if target already exists
            if target_path.exists():
                logger.warning(f"Target file already exists, skipping: {target_path}")
                self.stats['skipped_files'].append(str(source_path))
                return False
            
            # Copy file preserving original extension and content
            shutil.copy2(source_path, target_path)
            
            # Verify the copied file exists and has content
            if not target_path.exists() or target_path.stat().st_size == 0:
                logger.error(f"Failed to copy file: {target_path}")
                return False
            
            logger.debug(f"Copied: {source_path.name} -> {target_path.name}")
            return True
                
        except PermissionError as e:
            logger.error(f"Permission error copying {source_path}: {e}")
            self.stats['errors']['permission'].append(str(source_path))
            return False
        except OSError as e:
            logger.error(f"OS error copying {source_path}: {e}")
            self.stats['errors']['copy'].append((str(source_path), str(e)))
            return False
        except Exception as e:
            logger.error(f"Unexpected error copying {source_path}: {e}")
            self.stats['errors']['unexpected'].append((str(source_path), str(e)))
            return False
    
    def process_files(self, selected_files_by_subdir: Dict[str, List[Path]]) -> None:
        """
        Process the selected files into the run-specific subdirectory, organized by source subdirectory.
        
        Args:
            selected_files_by_subdir: Dictionary mapping source subdirectory names to lists of selected file paths
        """
        total_files = sum(len(files) for files in selected_files_by_subdir.values())
        logger.info(f"Starting to process {total_files} files...")
        logger.info(f"Files will be placed in run-specific directory: {self.run_target_dir}")
        logger.info(f"Files will be organized by source subdirectory within the run directory")
        
        # Check if we've already reached the target limit
        current_total = self.count_existing_files_in_target()
        if current_total >= self.target_count:
            logger.info(f"Target limit ({self.target_count}) already reached. Stopping processing.")
            return
        
        files_remaining = self.target_count - current_total
        files_processed_count = 0
        
        # Process files grouped by source subdirectory
        for source_subdir, selected_files in selected_files_by_subdir.items():
            # Create subdirectory within run directory for this source subdirectory
            subdir_target_dir = self.run_target_dir / source_subdir
            subdir_target_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {len(selected_files)} files from source subdirectory '{source_subdir}'")
            
            for i, source_path in enumerate(selected_files, 1):
                # Stop if we've reached the target limit
                if self.stats['files_copied'] >= files_remaining:
                    logger.info(f"Reached target limit of {self.target_count} files. Stopping processing.")
                    break
                
                files_processed_count += 1
                if files_processed_count % 100 == 0:
                    check_cancel_event(getattr(self, "_cancel_event", None))
                    logger.info(f"Progress: {files_processed_count}/{total_files} files processed")
                
                # Create target filename - preserve original extension
                # Place in run-specific subdirectory, organized by source subdirectory
                target_filename = source_path.name
                target_path = subdir_target_dir / target_filename
                
                # Handle filename conflicts within the subdirectory
                counter = 1
                original_target_path = target_path
                while target_path.exists():
                    stem = original_target_path.stem
                    suffix = original_target_path.suffix
                    target_path = subdir_target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Copy the file
                if self.copy_file(source_path, target_path):
                    self.stats['files_copied'] += 1
                
                self.stats['files_processed'] += 1
            
            # Break outer loop if we've reached the limit
            if self.stats['files_copied'] >= files_remaining:
                break
    
    def calculate_image_hash(self, image_path: Path) -> str:
        """
        Calculate a hash for an image based on its content.
        Uses simple MD5 of file content - much faster than PIL-based hashing.
        Since deduplication is handled separately, this is sufficient for avoiding reprocessing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MD5 hash string of the image content
        """
        try:
            md5_hash = hashlib.md5()
            with open(image_path, 'rb') as f:
                # Read in chunks for memory efficiency with large files
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {image_path}: {e}")
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
                import io
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
                image_hash = self.calculate_image_hash_pil(file_path)
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
                    image_hash = self.calculate_image_hash_pil(file_path)
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
    
    def perform_deduplication(self, raw_data_dir: Path = None) -> None:
        """
        Perform deduplication across class/staging directories under *raw_data_dir*.

        Bucket names come from ``data.class_names`` or :func:`~mb.data.class_layout.discover_class_names`
        (same rules as the rest of the pipeline), excluding ``rejected`` and ``small_images_review``.
        """
        if raw_data_dir is None:
            raw_data_dir = Path("raw_data")

        layout = layout_dict_for_discovery()
        bucket_names = discover_raw_data_bucket_names(
            raw_data_dir,
            explicit=layout["explicit"],
            class_qualifying_subdir=layout["class_qualifying_subdir"],
        )
        directories_to_check = [raw_data_dir / n for n in bucket_names]
        existing_directories = [d for d in directories_to_check if d.exists()]

        logger.info("=" * 60)
        logger.info("STARTING DEDUPLICATION PROCESS")
        logger.info("=" * 60)

        if not existing_directories:
            logger.warning("No class/staging directories found for deduplication")
            return

        logger.info(f"Found directories: {[d.name for d in existing_directories]}")

        # Step 1: Remove duplicates within each directory
        logger.info("\nStep 1: Removing duplicates within each directory...")
        for directory in existing_directories:
            removed_count = self.remove_duplicates_from_directory(directory)
            self.stats['duplicates_removed'] += removed_count

        # Step 2: Find duplicates across all directories (any bucket vs any bucket)
        logger.info("\nStep 2: Finding duplicates across all directories...")
        all_duplicates = self.find_duplicates_across_directories(existing_directories)
        self.stats['duplicates_found'] = len(all_duplicates)

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
        """Print a summary of the operation."""
        logger.info("=" * 60)
        logger.info("OPERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Run-specific target directory: {self.run_target_dir}")
        if self.rejected_dir:
            logger.info(f"Rejected directory (flat): {self.rejected_dir}")
        
        logger.info(f"\nTotal unprocessed files found: {self.stats['total_files_found']}")
        logger.info(f"Already-processed files (excluded): {self.stats['already_processed_count']}")
        logger.info(f"Files processed this run: {self.stats['files_processed']}")
        logger.info(f"Files successfully copied this run: {self.stats['files_copied']}")
        logger.info(f"Files skipped (already existed): {len(self.stats['skipped_files'])}")
        
        # Count totals across all subdirectories
        total_in_target = self.count_existing_files_in_target()
        logger.info(f"\nTotal files in target directory (all runs): {total_in_target}")
        logger.info(f"Target limit: {self.target_count}")
        logger.info(f"Remaining capacity: {max(0, self.target_count - total_in_target)}")
        
        logger.info(f"\nDuplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"Duplicate groups found: {self.stats['duplicates_found']}")
        
        if self.stats['errors']:
            logger.info("\nERRORS ENCOUNTERED:")
            for error_type, errors in self.stats['errors'].items():
                logger.info(f"  {error_type}: {len(errors)} errors")
                for error in errors[:5]:  # Show first 5 errors
                    logger.info(f"    - {error}")
                if len(errors) > 5:
                    logger.info(f"    ... and {len(errors) - 5} more")
        
        if self.stats['skipped_files']:
            logger.info(f"\nSkipped files (already existed): {len(self.stats['skipped_files'])}")
        
        logger.info(f"\nTarget directory: {self.target_dir}")
        logger.info(f"Target limit: {self.target_count}")
        logger.info(f"Files copied this run: {self.stats['files_copied']}")
        
        if total_in_target >= self.target_count:
            logger.info(f"Target limit reached! Total: {total_in_target}/{self.target_count}")
        else:
            logger.info(f"Target limit not yet reached. Current: {total_in_target}/{self.target_count}")
            if self.stats['files_copied'] > 0:
                logger.info(f"Added {self.stats['files_copied']} files this run.")
    
    def run(self, cancel_event: Optional[threading.Event] = None) -> bool:
        """Main execution method. *cancel_event* is checked periodically when set (e.g. GUI Cancel)."""
        self._cancel_event = cancel_event
        try:
            return self._run_impl()
        finally:
            self._cancel_event = None

    def _run_impl(self) -> bool:
        log_startup_info(logger, "Coherent image gathering process")
        logger.info(f"Source directory: {self.source_dir}")
        logger.info(f"Valid subdirectories: {self.valid_subdirs}")
        if self.subdir_weights:
            logger.info(f"Subdirectory weights: {self.subdir_weights}")
        logger.info(f"Target directory: {self.target_dir}")
        logger.info(f"Run-specific target directory: {self.run_target_dir}")
        if self.rejected_dir:
            logger.info(f"Rejected directory (flat): {self.rejected_dir}")
        logger.info(f"Target limit: {self.target_count}")
        logger.info(f"Run ID: {self.run_id}")
        
        check_cancel_event(self._cancel_event)
        
        # Load cache at startup
        self.load_cache()
        
        # Validate configuration
        if not self.validate_configuration():
            return False
        
        # Find all image files (grouped by subdirectory, excluding already-processed ones)
        files_by_subdir = self.find_image_files()
        check_cancel_event(self._cancel_event)
        if not files_by_subdir:
            logger.warning("No unprocessed image files found in the specified subdirectories")
            # This is not necessarily a failure - target limit might already be reached
            current_total = self.count_existing_files_in_target()
            if current_total >= self.target_count:
                logger.info("Target limit already reached. No action needed.")
                # Save cache before early return
                if self.cache_modified:
                    self.save_cache()
                self.print_summary()
                return True
            else:
                logger.error("No files available to process and target limit not reached")
                # Save cache before early return
                if self.cache_modified:
                    self.save_cache()
                return False
        
        # Select random files (up to target limit, with optional weighting)
        selected_files_by_subdir = self.select_random_files(files_by_subdir)
        check_cancel_event(self._cancel_event)
        if not selected_files_by_subdir:
            logger.info("No files were selected for processing (target limit may already be reached)")
            # Save cache before early return
            if self.cache_modified:
                self.save_cache()
            self.print_summary()
            return True
        
        # Process the files (organized by source subdirectory)
        self.process_files(selected_files_by_subdir)
        
        # Note: Deduplication is now handled by a separate script (deduplicate_images.py)
        # This allows it to be run independently and can take as long as needed
        
        # Print summary
        self.print_summary()
        
        # Save cache before completion
        if self.cache_modified:
            self.save_cache()
            logger.info("Cache saved at end of run")
        
        # Log completion
        total_in_target = self.count_existing_files_in_target()
        success = total_in_target >= self.target_count or self.stats['files_copied'] > 0
        message = f"Copied {self.stats['files_copied']} files this run. Total in target: {total_in_target}/{self.target_count}"
        log_completion_info(logger, success, message)
        
        return success


if __name__ == "__main__":
    from mb.cli import run_data_subcommand_cli

    sys.exit(run_data_subcommand_cli("gather"))
