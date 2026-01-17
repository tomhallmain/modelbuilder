"""
Storage utilities for checking external storage and drive information.
"""

import os
import sys
from pathlib import Path
import logging


def is_external_storage(path: Path) -> bool:
    """
    Returns True if the given path is on an external/removable drive.
    Works on Windows, Linux, and macOS. Falls back to False if undetectable.
    """
    path = Path(path).resolve()
    if sys.platform.startswith('win'):
        try:
            import win32file
            drive = os.path.splitdrive(str(path))[0] + '\\'
            drive_type = win32file.GetDriveType(drive)
            # DRIVE_REMOVABLE = 2, DRIVE_CDROM = 5, DRIVE_UNKNOWN = 1
            # DRIVE_FIXED = 3 (internal drives), DRIVE_REMOTE = 4 (network)
            return drive_type in (2, 5, 1)  # Only removable, CDROM, and unknown are considered external
        except ImportError:
            # pywin32 not installed - fall back to path-based detection
            drive = os.path.splitdrive(str(path))[0].lower()
            # Common external drive patterns on Windows
            external_patterns = ['usb', 'removable', 'external']
            return any(pattern in drive for pattern in external_patterns)
        except Exception:
            return False
    else:
        # Linux/macOS: check /proc/mounts or use psutil if available
        try:
            import psutil
            partitions = psutil.disk_partitions(all=False)
            for part in partitions:
                if path.as_posix().startswith(part.mountpoint):
                    # Removable devices often have 'removable' in opts or device path
                    if 'removable' in part.opts or '/media/' in part.mountpoint or '/Volumes/' in part.mountpoint:
                        return True
            return False
        except ImportError:
            # Fallback: check for /media or /Volumes in path
            if '/media/' in str(path) or '/Volumes/' in str(path):
                return True
            return False
        except Exception:
            return False


def check_target_external_storage(logger: logging.Logger, path: Path, override: bool = False):
    """
    Log a warning if the path is on external/removable storage, and remind user to use SSD for best performance.
    Returns True if external storage is detected and not overridden (process should fail).
    """
    if is_external_storage(path):
        logger.error(f"The directory {path} appears to be on external/removable storage. For best performance, move your data to an internal SSD.")
        if not override:
            logger.error("Process will fail. Use --allow-external-storage to override this check if you must use external storage.")
            return True  # Indicate process should fail
        else:
            logger.warning("External storage detected but override flag used. Performance may be significantly degraded.")
    else:
        logger.info(f"Data directory {path} appears to be on internal storage. [Recommended]")
    logger.info("Tip: Training on an internal SSD is strongly recommended for deep learning workflows with large datasets.")
    return False  # Process can continue


def check_same_drive(source_path: Path, target_path: Path) -> bool:
    """
    Check if source and target paths are on the same drive.
    Returns True if they are on the same drive, False otherwise.
    """
    source_drive = os.path.splitdrive(str(source_path.resolve()))[0]
    target_drive = os.path.splitdrive(str(target_path.resolve()))[0]
    return source_drive.lower() == target_drive.lower()


def log_drive_optimization_advice(logger: logging.Logger, source_path: Path, target_path: Path, override: bool = False):
    """
    Log advice about drive optimization for file operations.
    Note: Copy operations preserve source files, which is desired for dataset creation.
    """
    if check_same_drive(source_path, target_path):
        logger.info(f"Source ({source_path}) and target ({target_path}) are on the same drive.")
        logger.info("Copy operations will be faster on the same drive.")
    else:
        logger.info(f"Source ({source_path}) and target ({target_path}) are on different drives.")
        if not override:
            logger.info("Copy operations across drives may be slower than same-drive operations.")
            logger.info("Consider using the same drive for both source and target for better performance.")
        else:
            logger.info("Different drives detected but override flag used. Using copy operations.")
