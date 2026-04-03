#!/usr/bin/env python3
"""
Build train/test splits under ``data_dir`` from raw class folders (ImageFolder-style layout).

:class:`DatasetCreator` resolves class names from pipeline YAML and/or discovery
(see :mod:`mb.data.class_layout`), copies validated files with hash-based names,
optionally balances classes, and forms a per-class test split. PIL validates images
and enforces size bounds for the image pipeline.

**CLI:** ``mb data create-dataset``; ``python -m mb.data.dataset`` delegates via
:func:`mb.cli.run_data_subcommand_cli`. Pipeline: gather → convert → this step;
requires a unified snapshot from conversion where applicable.

Implementation notes:
- Efficient image validation using PIL
- Randomization for split creation
- Combined logic for maintainability
"""

import sys
import shutil
import random
import threading
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

# Image processing imports
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise ImportError("Error: PIL/Pillow not available. Image validation is required.")

# Import centralized logging configuration
from mb.utils.logging_setup import log_completion_info, log_startup_info, setup_logging
from mb.utils.translations import _
from mb.cancellation import check_cancel_event
from mb.utils.snapshot import (
    UnifiedSnapshot,
    calculate_file_hash,
    find_unified_snapshot,
    preload_gather_cache,
    save_unified_snapshot,
)
from mb.data.class_layout import (
    discover_class_names,
    normalize_qualifying_subdir,
    resolve_class_media_dir,
)
from mb.pipeline_config import get_pipeline_config
from mb.space_estimate import check_create_dataset_allowed

# Configure logging
logger = setup_logging(script_name="create_datasets")

DEFAULT_TEST_PER_CLASS = 1000  # Default test-split size per class (CLI / library; pipeline YAML may override)
MIN_FILE_SIZE = 6 * 1024  # 6KB minimum
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB maximum

class DatasetCreator:
    """Handles the creation of training and test datasets."""
    
    def __init__(
        self,
        raw_data_dir: Path,
        data_dir: Path,
        test_per_class: int = DEFAULT_TEST_PER_CLASS,
        balance_train: bool = False,
        max_train_per_class: Optional[int] = None,
        run_id: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        class_qualifying_subdir: Optional[str] = None,
        skip_space_check: bool = False,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.data_dir = Path(data_dir)
        self.test_per_class = test_per_class
        self.balance_train = balance_train  # If True, balance training set to smallest class
        self.max_train_per_class = max_train_per_class  # Optional: limit per class (None = no limit)
        self.run_id = run_id  # Optional run ID for unified snapshot
        self._class_names_override = class_names
        self._class_qualifying_subdir_override = class_qualifying_subdir
        self._skip_space_check = skip_space_check
        self._class_names: List[str] = []
        self._class_qualifying_subdir: Optional[str] = None

        # Directory paths
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
        # Review directory for invalid-sized images
        self.review_dir = self.data_dir / "invalid_size_review"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'raw_files_found': defaultdict(int),
            'files_copied': defaultdict(int),
            'files_removed_corrupted': defaultdict(int),
            'files_moved_size': defaultdict(int),
            'test_files_moved': defaultdict(int),
            'final_train_counts': defaultdict(int),
            'final_test_counts': defaultdict(int)
        }
        
        # Ensure directories exist
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified snapshot (will be loaded)
        self.unified_snapshot: Optional[UnifiedSnapshot] = None

    def _resolve_class_names(self) -> bool:
        """Set :attr:`_class_names` and :attr:`_class_qualifying_subdir` from overrides and pipeline."""
        pc = get_pipeline_config()
        qual = self._class_qualifying_subdir_override
        if qual is None:
            qual = pc.get("data.class_qualifying_subdir")
        self._class_qualifying_subdir = normalize_qualifying_subdir(qual)

        explicit = self._class_names_override
        if explicit is None:
            ex = pc.get("data.class_names")
            explicit = ex if isinstance(ex, list) else None

        names = discover_class_names(
            self.raw_data_dir,
            explicit=explicit,
            class_qualifying_subdir=self._class_qualifying_subdir,
        )
        self._class_names = names
        if not names:
            logger.error(
                "No class directories found under %s (check data.class_names / data.class_qualifying_subdir)",
                self.raw_data_dir,
            )
            return False
        logger.info("Class folders for dataset creation: %s", names)
        return True

    def validate_image(self, image_path: Path) -> bool:
        """Validate image using PIL - much more efficient than ImageMagick convert."""
        try:
            with Image.open(image_path) as img:
                # Try to access image data to ensure it's valid
                img.verify()
                return True
        except (UnidentifiedImageError, OSError, Exception):
            return False
    
    
    def copy_files_to_train(self) -> None:
        """Copy files from raw_data to train directory with hash-based filenames."""
        logger.info("Copying images to training folder...")
        
        # Preload gather cache for faster hash lookups
        cache_loaded = preload_gather_cache(self.raw_data_dir)
        if cache_loaded:
            logger.info("Gather cache loaded successfully - hash lookups will be faster")
        
        for class_name in self._class_names:
            check_cancel_event(getattr(self, "_cancel_event", None))
            raw_class_dir = self.raw_data_dir / class_name
            train_class_dir = self.train_dir / class_name
            
            if not raw_class_dir.exists():
                logger.warning(f"Raw data directory for class '{class_name}' does not exist: {raw_class_dir}")
                continue
                
            train_class_dir.mkdir(exist_ok=True)

            media_dir = resolve_class_media_dir(raw_class_dir, self._class_qualifying_subdir)
            if media_dir is None or not media_dir.exists():
                logger.warning(
                    "No image source directory for class '%s' (expected layout under %s); skipping",
                    class_name,
                    raw_class_dir,
                )
                continue

            image_files = list(media_dir.glob("*.jpg")) + list(media_dir.glob("*.jpeg"))
            self.stats['raw_files_found'][class_name] = len(image_files)
            
            logger.info(f"Found {len(image_files)} images for class '{class_name}'")
            
            for idx, image_file in enumerate(image_files, 1):
                # Log progress every 1000 files
                if idx % 1000 == 0:
                    check_cancel_event(getattr(self, "_cancel_event", None))
                    logger.info(f"Processing {class_name}: {idx}/{len(image_files)} images ({idx*100//len(image_files)}%)")
                
                # No .is_file() check needed - glob() only returns files
                # Calculate both MD5 (for matching) and SHA256 (for filename)
                # Pass raw_data_dir to enable gather cache lookup
                md5_hash = calculate_file_hash(image_file, algorithm='md5', raw_data_dir=self.raw_data_dir, unified_snapshot=self.unified_snapshot, logger=logger)
                sha256_hash = calculate_file_hash(image_file, algorithm='sha256', raw_data_dir=self.raw_data_dir, logger=logger)
                
                # Skip if hash calculation failed
                if md5_hash is None or sha256_hash is None:
                    logger.warning(f"Failed to calculate hash for {image_file}, skipping")
                    continue
                
                # Generate hash-based filename (using SHA256)
                target_path = train_class_dir / f"{sha256_hash}.jpg"
                
                # Copy file
                shutil.copy2(image_file, target_path)
                self.stats['files_copied'][class_name] += 1
                
                # Record in unified snapshot - method will find the record itself
                if self.unified_snapshot:
                    self.unified_snapshot.add_dataset_image(
                        class_name=class_name,
                        converted_path=str(image_file.relative_to(self.raw_data_dir)),
                        converted_basename=image_file.name,
                        converted_md5=md5_hash,
                        converted_sha256=sha256_hash,
                        final_path=f"train/{class_name}/{sha256_hash}.jpg",
                        final_basename=f"{sha256_hash}.jpg"
                    )
            
            # Log completion for this class
            logger.info(f"Completed processing {class_name}: {len(image_files)} images processed")
    
    def remove_corrupted_images(self) -> None:
        """Remove corrupted images using efficient PIL validation."""
        logger.info("Removing corrupted images...")
        
        for class_name in self._class_names:
            check_cancel_event(getattr(self, "_cancel_event", None))
            train_class_dir = self.train_dir / class_name
            if not train_class_dir.exists():
                continue
                
            image_files = list(train_class_dir.glob("*.jpg"))
            logger.info(f"Validating {len(image_files)} images for class '{class_name}'")
            removed_count = 0
            
            for idx, image_file in enumerate(image_files, 1):
                # Log progress every 1000 files
                if idx % 1000 == 0:
                    check_cancel_event(getattr(self, "_cancel_event", None))
                    logger.info(f"Validating {class_name}: {idx}/{len(image_files)} images ({idx*100//len(image_files)}%)")
                
                if not self.validate_image(image_file):
                    logger.debug(f"Removing corrupted image: {image_file}")
                    # Remove from snapshot
                    final_path = f"train/{class_name}/{image_file.name}"
                    if self.unified_snapshot:
                        self.unified_snapshot.remove_dataset_image(final_path)
                    image_file.unlink()
                    removed_count += 1
            
            self.stats['files_removed_corrupted'][class_name] = removed_count
            logger.info(f"Completed validation for {class_name}: {len(image_files)} checked, {removed_count} corrupted images removed")
    
    def remove_invalid_sized_images(self) -> None:
        """Move images with invalid file sizes to review directory."""
        logger.info("Checking images for invalid file sizes...")
        
        for class_name in self._class_names:
            train_class_dir = self.train_dir / class_name
            if not train_class_dir.exists():
                continue
                
            image_files = list(train_class_dir.glob("*.jpg"))
            moved_count = 0
            
            for image_file in image_files:
                file_size = image_file.stat().st_size
                
                if file_size < MIN_FILE_SIZE or file_size > MAX_FILE_SIZE:
                    # Determine reason for moving
                    if file_size < MIN_FILE_SIZE:
                        reason = f"too small ({file_size} bytes < {MIN_FILE_SIZE} bytes)"
                    else:
                        reason = f"too large ({file_size} bytes > {MAX_FILE_SIZE} bytes)"
                    
                    logger.warning(f"Image {image_file.name} in class '{class_name}' is {reason}, moving to review directory")
                    
                    # Remove from snapshot first (as if removed, user can review and decide)
                    final_path = f"train/{class_name}/{image_file.name}"
                    if self.unified_snapshot:
                        self.unified_snapshot.remove_dataset_image(final_path)
                    
                    # Create target path preserving relative structure
                    relative_path = image_file.relative_to(self.train_dir)
                    target_path = self.review_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Handle filename conflicts
                    counter = 1
                    original_target = target_path
                    while target_path.exists():
                        stem = original_target.stem
                        suffix = original_target.suffix
                        target_path = original_target.parent / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    # Move file to review directory
                    shutil.move(str(image_file), str(target_path))
                    moved_count += 1
                    logger.debug(f"Moved invalid-sized image: {image_file.name} ({reason}) -> {target_path}")
            
            self.stats['files_moved_size'][class_name] = moved_count
            if moved_count > 0:
                logger.info(f"Moved {moved_count} invalid-sized images from class '{class_name}' to review directory")
            else:
                logger.info(f"No invalid-sized images found in class '{class_name}'")
    
    def balance_training_set(self) -> None:
        """
        Balance training set by randomly sampling from each class.
        If balance_train=True: balances to smallest class size
        If max_train_per_class is set: limits each class to that number
        """
        logger.info("Balancing training set...")
        
        # First, count current images per class
        class_counts = {}
        class_images = {}
        for class_name in self._class_names:
            train_class_dir = self.train_dir / class_name
            if train_class_dir.exists():
                images = list(train_class_dir.glob("*.jpg"))
                class_counts[class_name] = len(images)
                class_images[class_name] = images
            else:
                class_counts[class_name] = 0
                class_images[class_name] = []
        
        # Determine target count
        if self.max_train_per_class:
            target_count = self.max_train_per_class
            logger.info(f"Limiting each class to {target_count} images (max_train_per_class)")
        elif self.balance_train:
            # Balance to smallest class
            min_count = min(class_counts.values()) if class_counts.values() else 0
            target_count = min_count
            logger.info(f"Balancing training set to smallest class size: {target_count} images per class")
        else:
            # Shouldn't reach here, but return early if neither option is set
            return
        
        if target_count == 0:
            logger.warning("Cannot balance: no images found in any class")
            return
        
        # Log current distribution
        logger.info("Current training set distribution:")
        for class_name in self._class_names:
            current_count = class_counts[class_name]
            logger.info(f"  {class_name}: {current_count} images")
        
        # Balance each class
        removed_count = defaultdict(int)
        for class_name in self._class_names:
            train_class_dir = self.train_dir / class_name
            if not train_class_dir.exists():
                continue
            
            images = class_images[class_name]
            current_count = len(images)
            
            if current_count > target_count:
                # Randomly sample to target count
                images_to_keep = random.sample(images, target_count)
                images_to_remove = set(images) - set(images_to_keep)
                
                # Remove excess images
                for image_file in images_to_remove:
                    # Remove from snapshot
                    final_path = f"train/{class_name}/{image_file.name}"
                    if self.unified_snapshot:
                        self.unified_snapshot.remove_dataset_image(final_path)
                    image_file.unlink()
                    removed_count[class_name] += 1
                
                logger.info(f"Balanced {class_name}: kept {target_count}, removed {len(images_to_remove)}")
            elif current_count < target_count:
                logger.warning(f"Class '{class_name}' has only {current_count} images, cannot reach target of {target_count}")
            else:
                logger.info(f"Class '{class_name}' already at target count: {current_count}")
        
        # Log summary
        total_removed = sum(removed_count.values())
        if total_removed > 0:
            logger.info(f"Balancing complete: removed {total_removed} images total")
            logger.info("Final balanced training set distribution:")
            for class_name in self._class_names:
                train_class_dir = self.train_dir / class_name
                if train_class_dir.exists():
                    final_count = len(list(train_class_dir.glob("*.jpg")))
                    logger.info(f"  {class_name}: {final_count} images")
        else:
            logger.info("No balancing needed - all classes already balanced")
    
    def create_test_dataset(self) -> None:
        """Create test dataset by randomly moving files from train to test."""
        logger.info(f"Creating test dataset with {self.test_per_class} items per class...")
        
        for class_name in self._class_names:
            train_class_dir = self.train_dir / class_name
            test_class_dir = self.test_dir / class_name
            
            if not train_class_dir.exists():
                logger.warning(f"Train directory for class '{class_name}' does not exist")
                continue
                
            test_class_dir.mkdir(exist_ok=True)
            
            # Get all image files in train directory
            train_images = list(train_class_dir.glob("*.jpg"))
            
            if len(train_images) < self.test_per_class:
                logger.warning(f"Not enough images in class '{class_name}': {len(train_images)} available, {self.test_per_class} needed")
                # Use all available images for test
                selected_images = train_images
            else:
                # Randomly select images for test set
                selected_images = random.sample(train_images, self.test_per_class)
            
            # Move selected images to test directory and update snapshot
            for image_file in selected_images:
                target_path = test_class_dir / image_file.name
                shutil.move(str(image_file), str(target_path))
                self.stats['test_files_moved'][class_name] += 1
                
                # Update snapshot: change final_path from train to test
                old_final_path = f"train/{class_name}/{image_file.name}"
                if self.unified_snapshot:
                    self.unified_snapshot.update_dataset_split(old_final_path, 'test')
            
            logger.info(f"Moved {len(selected_images)} images to test set for class '{class_name}'")
    
    def save_dataset_snapshot(self) -> None:
        """Save unified snapshot to JSON file."""
        if self.unified_snapshot:
            # Update data directory if not set
            if not self.unified_snapshot.data_directory:
                self.unified_snapshot.data_directory = str(self.data_dir)
            
            # Save unified snapshot
            snapshot_path = save_unified_snapshot(self.unified_snapshot, self.data_dir, logger)
            if snapshot_path:
                logger.info(f"Unified snapshot updated with dataset creation data (run_id: {self.unified_snapshot.run_id})")
    
    def count_final_files(self) -> None:
        """Count final number of files in train and test directories."""
        logger.info("Counting final files...")
        
        for class_name in self._class_names:
            train_class_dir = self.train_dir / class_name
            test_class_dir = self.test_dir / class_name
            
            if train_class_dir.exists():
                train_count = len(list(train_class_dir.glob("*.jpg")))
                self.stats['final_train_counts'][class_name] = train_count
            else:
                self.stats['final_train_counts'][class_name] = 0
                
            if test_class_dir.exists():
                test_count = len(list(test_class_dir.glob("*.jpg")))
                self.stats['final_test_counts'][class_name] = test_count
            else:
                self.stats['final_test_counts'][class_name] = 0
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of the operation."""
        logger.info("=" * 80)
        logger.info("DATASET CREATION SUMMARY")
        logger.info("=" * 80)
        
        logger.info("\nRAW DATA STATISTICS:")
        for class_name in self._class_names:
            count = self.stats['raw_files_found'][class_name]
            logger.info(f"  {class_name}: {count} files found")
        
        logger.info("\nPROCESSING STATISTICS:")
        for class_name in self._class_names:
            copied = self.stats['files_copied'][class_name]
            corrupted = self.stats['files_removed_corrupted'][class_name]
            invalid_size = self.stats['files_moved_size'][class_name]
            logger.info(f"  {class_name}: {copied} copied, {corrupted} corrupted removed, {invalid_size} invalid size moved to review")
        
        logger.info("\nTEST SET CREATION:")
        for class_name in self._class_names:
            moved = self.stats['test_files_moved'][class_name]
            logger.info(f"  {class_name}: {moved} files moved to test set")
        
        logger.info("\nFINAL DATASET COUNTS:")
        logger.info("Training set:")
        for class_name in self._class_names:
            count = self.stats['final_train_counts'][class_name]
            logger.info(f"  {class_name}: {count} files")
        
        logger.info("Test set:")
        for class_name in self._class_names:
            count = self.stats['final_test_counts'][class_name]
            logger.info(f"  {class_name}: {count} files")
        
        # Calculate totals
        total_train = sum(self.stats['final_train_counts'].values())
        total_test = sum(self.stats['final_test_counts'].values())
        logger.info(f"\nTOTALS:")
        logger.info(f"  Training set: {total_train} files")
        logger.info(f"  Test set: {total_test} files")
        logger.info(f"  Combined: {total_train + total_test} files")
        
        logger.info(f"\nDirectories created:")
        logger.info(f"  Training: {self.train_dir}")
        logger.info(f"  Test: {self.test_dir}")
    
    def run(
        self,
        cancel_event: Optional[threading.Event] = None,
        *,
        skip_space_check: Optional[bool] = None,
    ) -> bool:
        """Main execution method. *cancel_event* is checked between steps and during long file loops."""
        self._cancel_event = cancel_event
        if skip_space_check is not None:
            self._skip_space_check = skip_space_check
        try:
            return self._run_impl()
        finally:
            self._cancel_event = None

    def _run_impl(self) -> bool:
        log_startup_info(logger, "Dataset creation process")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Output data directory: {self.data_dir}")
        logger.info(f"Test split size per class: {self.test_per_class}")
        
        check_cancel_event(self._cancel_event)
        
        # Validate raw data directory
        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory does not exist: {self.raw_data_dir}")
            return False

        if not self._resolve_class_names():
            return False

        # Load unified snapshot
        logger.info("Loading unified snapshot...")
        search_paths = [self.raw_data_dir, self.raw_data_dir.parent, self.data_dir]
        for class_name in self._class_names:
            search_paths.append(self.raw_data_dir / class_name)
        
        self.unified_snapshot = find_unified_snapshot(search_paths, run_id=self.run_id, logger=logger)
        if self.unified_snapshot:
            logger.info(f"Loaded unified snapshot with run_id: {self.unified_snapshot.run_id}")
            logger.info(f"Total images in snapshot: {len(self.unified_snapshot.images)}")
            if not self.unified_snapshot.data_directory:
                self.unified_snapshot.data_directory = str(self.data_dir)
        else:
            logger.error("No unified snapshot found! Run ``mb data convert`` (or equivalent) first to create a snapshot.")
            logger.error("Or provide a valid --run-id if the snapshot exists elsewhere.")
            return False

        allowed, _ = check_create_dataset_allowed(
            self.raw_data_dir,
            self.data_dir,
            snapshot=self.unified_snapshot,
            skip_space_check=self._skip_space_check,
        )
        if not allowed:
            logger.error(
                "Insufficient disk space on the output data directory (heuristic). "
                "Free space or pass skip_space_check / --skip-space-check if you accept the risk."
            )
            return False
        
        check_cancel_event(self._cancel_event)
        
        # Step 1: Copy files to train directory with hash-based filenames
        self.copy_files_to_train()
        
        check_cancel_event(self._cancel_event)
        
        # Step 2: Remove corrupted images
        self.remove_corrupted_images()
        
        check_cancel_event(self._cancel_event)
        
        # Step 3: Remove invalid-sized images
        self.remove_invalid_sized_images()
        
        check_cancel_event(self._cancel_event)
        
        # Step 3.5: Balance training set if requested (before test set creation)
        if self.balance_train or self.max_train_per_class:
            self.balance_training_set()
        
        check_cancel_event(self._cancel_event)
        
        # Step 4: Create test dataset by moving files
        self.create_test_dataset()
        
        check_cancel_event(self._cancel_event)
        
        # Step 5: Count final files
        self.count_final_files()
        
        # Step 6: Save dataset snapshot
        logger.info("Saving dataset snapshot...")
        self.save_dataset_snapshot()
        
        # Step 7: Print summary
        self.print_summary()
        
        # Log completion
        total_train = sum(self.stats['final_train_counts'].values())
        total_test = sum(self.stats['final_test_counts'].values())
        message = f"Created {total_train} training and {total_test} test images"
        log_completion_info(logger, True, message)
        
        return True


def confirm_user_action(logger, args):
    # Platform-specific warning covering cases of potential external drive misdetection
    logger.info(_("Source and target directories are on the same drive."))
    logger.info(
        _("The script will use copy operations (not move) to preserve source files.")
    )
    
    import platform
    system_drive = None
    
    if platform.system() == "Windows":
        try:
            import win32api
            system_drive = win32api.GetSystemDirectory()[:2]  # Get system drive (e.g., "C:")
        except ImportError:
            # If win32api fails, assume C: on Windows
            system_drive = "C:"
    
    source_drive = Path(args.raw_data_dir).drive
    
    # If we're on the system drive, no need to confirm - user knows what they're doing
    if system_drive and source_drive.upper() == system_drive.upper():
        logger.info(
            _("Source drive {drive} is the system drive. Proceeding without confirmation.").format(
                drive=source_drive
            )
        )
        return True
    
    # Platform-specific warning about potential external drive misdetection
    if platform.system() == "Windows":
        if system_drive and source_drive.upper() != system_drive.upper():
            logger.warning(
                _("NOTE: Source drive {src} is different from system drive {sys}.").format(
                    src=source_drive, sys=system_drive
                )
            )
            logger.warning(
                _(
                    "This may indicate an external drive that failed detection. Consider whether to proceed."
                )
            )
        else:
            logger.warning(
                _(
                    "NOTE: Unable to determine system drive. If this is not your main system drive,"
                )
            )
            logger.warning(
                _(
                    "it may indicate an external drive that failed detection. Consider whether to proceed."
                )
            )
    else:
        logger.warning(
            _("NOTE: If this is not your main system drive, it may indicate an external drive")
        )
        logger.warning(_("that failed detection. Consider whether to proceed."))
    
    response = input(_("Continue with same-drive operation? (y/N): ")).strip().lower()
    if response not in ['y', 'yes']:
        logger.info(_("Operation cancelled by user."))
        return False
    return True


if __name__ == "__main__":
    from mb.cli import run_data_subcommand_cli

    sys.exit(run_data_subcommand_cli("create-dataset"))