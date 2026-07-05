"""
Training-time updates to the unified image snapshot.

:class:`~mb.utils.snapshot.UnifiedSnapshot` is produced during convert / dataset
creation; :func:`update_training_snapshot` walks ``train/`` and ``test/`` under
``data_dir`` and attaches ``training`` metadata for each file that matches an
existing dataset record.
"""

from pathlib import Path
from typing import Callable, Optional
import threading

from mb.cancellation import check_cancel_event
from mb.data.file_types import configured_media_suffixes
from mb.utils.logging_setup import get_logger
from mb.utils.snapshot import UnifiedSnapshot, calculate_file_hash

logger = get_logger(__name__)


def update_training_snapshot(
    data_dir: Path,
    unified_snapshot: UnifiedSnapshot,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[Callable[[str, Optional[float]], None]] = None,
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
    
    def scan_class(
        class_dir: Path,
        split: str,
        class_index: int,
        total_classes: int,
    ) -> None:
        """Scan a directory and populate snapshot data."""
        check_cancel_event(cancel_event)
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
            return

        # Process each image
        for i, image_path in enumerate(image_files, 1):
            check_cancel_event(cancel_event)
            # Log progress every 1000 files
            if i % 1000 == 0:
                logger.info(
                    f"Processing {split}/{class_name}: {i}/{len(image_files)} images "
                    f"({i*100//len(image_files)}%)"
                )
            if progress_cb is not None and total_classes > 0:
                class_start = class_index / total_classes
                class_span = 1.0 / total_classes
                class_frac = i / max(len(image_files), 1)
                overall = min(class_start + class_span * class_frac, 1.0)
                progress_cb(
                    f"Updating snapshot: {split}/{class_name} — {i}/{len(image_files)} images",
                    overall,
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
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()] if train_dir.exists() else []
    test_classes = [d for d in test_dir.iterdir() if d.is_dir()] if test_dir.exists() else []
    total_classes = len(train_classes) + len(test_classes)
    class_idx = 0
    for class_dir in train_classes:
        scan_class(class_dir, 'train', class_idx, total_classes)
        class_idx += 1
    for class_dir in test_classes:
        scan_class(class_dir, 'test', class_idx, total_classes)
        class_idx += 1
    
    summary = unified_snapshot.to_dict().get('summary', {})
    train_total = summary.get('training_train_count', 0)
    test_total = summary.get('training_test_count', 0)
    logger.info(
        f"Training snapshot updated: {train_total} train images, {test_total} test images"
    )
