"""
Snapshot integration for training.

This module provides utilities for updating unified snapshots during training.
"""

from pathlib import Path
from typing import Optional
import logging

from mb.data.file_types import configured_media_suffixes
from mb.utils.snapshot import (
    UnifiedSnapshot,
    find_unified_snapshot,
    save_unified_snapshot,
    calculate_file_hash,
    preload_gather_cache,
)

logger = logging.getLogger(__name__)


def update_training_snapshot(
    data_dir: Path,
    unified_snapshot: UnifiedSnapshot
) -> None:
    """
    Update unified snapshot with training stage data.
    
    Scans train and test directories and adds image info to unified snapshot.
    This mirrors the functionality from the original train_model.py script.
    
    Args:
        data_dir: Root data directory containing train/ and test/ subdirectories
        unified_snapshot: UnifiedSnapshot instance to update
    """
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    def scan_directory(directory: Path, split: str) -> None:
        """Scan a directory and populate snapshot data."""
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return
        
        # Get all class subdirectories
        class_dirs = [d for d in directory.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            logger.info(f"Scanning {split}/{class_name}...")
            
            # Find all image files recursively
            image_files = []
            for ext in configured_media_suffixes():
                image_files.extend(class_dir.rglob(f'*{ext}'))
                image_files.extend(class_dir.rglob(f'*{ext.upper()}'))
            
            # Remove duplicates and sort
            image_files = sorted(set(image_files))
            
            logger.info(f"Found {len(image_files)} images in {split}/{class_name}")
            
            if not image_files:
                logger.warning(f"No images found in {split}/{class_name}, skipping")
                continue
            
            # Process each image
            for i, image_path in enumerate(image_files, 1):
                # Log progress every 1000 files
                if i % 1000 == 0:
                    logger.info(
                        f"Processing {split}/{class_name}: {i}/{len(image_files)} images "
                        f"({i*100//len(image_files)}%)"
                    )
                
                try:
                    # Get relative path from data directory (needed for snapshot lookup)
                    relative_path = image_path.relative_to(data_dir)
                    
                    # Calculate hash - use unified snapshot to map back to original path in gather cache
                    image_hash = calculate_file_hash(
                        image_path,
                        algorithm='md5',
                        unified_snapshot=unified_snapshot,
                        relative_path=str(relative_path),
                        logger=logger
                    )
                    if image_hash is None:
                        logger.warning(f"Failed to hash {image_path}, skipping")
                        continue
                    
                    # Add training info - the method will find the record and update it
                    unified_snapshot.add_training_image(
                        split=split,
                        class_name=class_name,
                        path=str(relative_path),
                        hash=image_hash,
                        basename=image_path.name
                    )
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
            
            # Log completion for this class
            logger.info(
                f"Completed scanning {split}/{class_name}: {len(image_files)} images processed"
            )
    
    # Scan train and test directories
    logger.info("Updating unified snapshot with training data...")
    scan_directory(train_dir, 'train')
    scan_directory(test_dir, 'test')
    
    summary = unified_snapshot.to_dict().get('summary', {})
    train_total = summary.get('training_train_count', 0)
    test_total = summary.get('training_test_count', 0)
    logger.info(
        f"Training snapshot updated: {train_total} train images, {test_total} test images"
    )
