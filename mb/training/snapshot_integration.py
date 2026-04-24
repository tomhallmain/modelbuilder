"""
Training-time updates to the unified image snapshot.

:class:`~mb.utils.snapshot.UnifiedSnapshot` is produced during convert / dataset
creation; :func:`update_training_snapshot` walks ``train/`` and ``test/`` under
``data_dir`` and attaches ``training`` metadata for each file that matches an
existing dataset record.
"""

from pathlib import Path
from typing import Optional
import logging

from mb.data.file_types import configured_media_suffixes
from mb.utils.snapshot import UnifiedSnapshot, calculate_file_hash

logger = logging.getLogger(__name__)


def update_training_snapshot(
    data_dir: Path,
    unified_snapshot: UnifiedSnapshot
) -> None:
    """
    Scan ``data_dir/train`` and ``data_dir/test`` and populate the ``training`` field
    on matching snapshot records (see :meth:`~mb.utils.snapshot.UnifiedSnapshot.add_training_image`).
    
    Args:
        data_dir: Root data directory containing train/ and test/ subdirectories
        unified_snapshot: UnifiedSnapshot instance to update
    """
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    media_suffixes = tuple(configured_media_suffixes())
    
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
            for ext in media_suffixes:
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
