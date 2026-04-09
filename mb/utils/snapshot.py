#!/usr/bin/env python3
"""
Snapshot utilities for tracking images through the processing pipeline.
Handles creation, loading, and matching of image snapshots across different stages.
"""

import hashlib
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from mb.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ``step_errors`` keys: use :class:`~mb.models.types.ModelBuildStepCommand` / ``.value`` for
# ``mb data`` steps and :class:`~mb.utils.constants.ModelBuilderTaskType` / ``.value`` for training.


def register_step_error(
    snapshot: "UnifiedSnapshot",
    step: str,
    run_started_at: str,
    message: str,
) -> None:
    """
    Append one error line for a pipeline *step* and invocation (*run_started_at* ISO timestamp).

    Multiple invocations of the same step append under separate *run_started_at* keys.
    """
    text = str(message).strip()
    if not text:
        return
    step_key = str(step).strip()
    ts_key = str(run_started_at).strip()
    if not step_key or not ts_key:
        return
    if step_key not in snapshot.step_errors:
        snapshot.step_errors[step_key] = {}
    if ts_key not in snapshot.step_errors[step_key]:
        snapshot.step_errors[step_key][ts_key] = []
    snapshot.step_errors[step_key][ts_key].append(text)
    snapshot.last_updated = datetime.now().isoformat()


def set_step_errors_for_invocation(
    snapshot: "UnifiedSnapshot",
    step: str,
    run_started_at: str,
    messages: Sequence[str],
) -> None:
    """
    Set the full error list for one step invocation (replaces any prior list for that key).

    Use an empty *messages* to record a completed invocation with no errors.
    """
    step_key = str(step).strip()
    ts_key = str(run_started_at).strip()
    if not step_key or not ts_key:
        return
    if step_key not in snapshot.step_errors:
        snapshot.step_errors[step_key] = {}
    cleaned = [str(m).strip() for m in messages if str(m).strip()]
    snapshot.step_errors[step_key][ts_key] = cleaned
    snapshot.last_updated = datetime.now().isoformat()


def flatten_convert_stats_errors(stats_errors: Any) -> List[str]:
    """Turn :class:`collections.defaultdict` lists from convert into snapshot-ready strings."""
    out: List[str] = []
    if not stats_errors:
        return out
    try:
        items = stats_errors.items()
    except AttributeError:
        return out
    for category, entries in items:
        if not entries:
            continue
        for item in entries:
            out.append(f"{category}: {item!r}")
    return out


# Backward-friendly alias (see pipeline docs).
register_error = register_step_error


def _posix_rel(p: Optional[str]) -> Optional[str]:
    """Normalize relative path strings for comparisons (Windows ``\\\\`` vs ``/``)."""
    if p is None:
        return None
    return str(p).replace("\\", "/")


def _coerce_loaded_step_errors(raw: Any) -> Dict[str, Dict[str, List[str]]]:
    """Validate ``step_errors`` JSON into nested step → timestamp → messages."""
    out: Dict[str, Dict[str, List[str]]] = {}
    if not isinstance(raw, dict):
        return out
    for step, inv_map in raw.items():
        sk = str(step).strip()
        if not sk or not isinstance(inv_map, dict):
            continue
        out[sk] = {}
        for ts_key, msgs in inv_map.items():
            tk = str(ts_key).strip()
            if not tk:
                continue
            if isinstance(msgs, list):
                out[sk][tk] = [str(m) for m in msgs if m is not None]
            else:
                out[sk][tk] = []
    return out


# Global cache for gather cache (lazy-loaded)
_gather_cache: Optional[Dict[str, str]] = None
_gather_cache_path: Optional[Path] = None


def generate_run_id() -> str:
    """
    Generate a unique run ID for tracking a complete pipeline run.
    
    Returns:
        Run ID string in format: YYYYMMDD_HHMMSS_<short_uuid>
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def preload_gather_cache(raw_data_dir: Optional[Path] = None) -> bool:
    """
    Preload the gather cache from :mod:`mb.data.gather` if available.
    Cache is stored at raw_data/.gather_cache.pkl
    
    This should be called once before processing many files to avoid
    repeated cache loading checks in calculate_file_hash.
    
    Args:
        raw_data_dir: Optional raw_data directory path. If None, tries to infer from common locations.
        
    Returns:
        True if cache was loaded successfully, False otherwise
    """
    global _gather_cache, _gather_cache_path
    
    # Return True if already loaded
    if _gather_cache is not None:
        return True
    
    # Try to find cache file
    cache_paths = []
    if raw_data_dir:
        cache_paths.append(Path(raw_data_dir) / ".gather_cache.pkl")
    # Also try common locations
    cache_paths.extend([
        Path("raw_data/.gather_cache.pkl"),
        Path("../raw_data/.gather_cache.pkl"),
    ])
    
    for cache_path in cache_paths:
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    loaded_cache = pickle.load(f)
                
                # Handle both old format (nested dict) and new format (direct hash)
                if loaded_cache:
                    first_value = next(iter(loaded_cache.values()))
                    if isinstance(first_value, dict):
                        # Old format: migrate to new format
                        new_cache = {}
                        for file_path, cache_entry in loaded_cache.items():
                            if isinstance(cache_entry, dict):
                                hash_value = cache_entry.get('hash')
                                if hash_value:
                                    new_cache[file_path] = hash_value
                        _gather_cache = new_cache
                    else:
                        # New format: use directly
                        _gather_cache = loaded_cache
                
                _gather_cache_path = cache_path
                return True
            except Exception:
                # If loading fails, continue to next path
                continue
    
    # Cache not found or failed to load
    _gather_cache = {}
    return False


def calculate_file_hash(file_path: Path, algorithm: str = 'md5', raw_data_dir: Optional[Path] = None, 
                        unified_snapshot: Optional['UnifiedSnapshot'] = None, relative_path: Optional[str] = None,
                        logger=None) -> Optional[str]:
    """
    Calculate hash of a file.
    For MD5 hashes, checks the gather cache first to avoid recalculating.
    Can use unified_snapshot to map training/dataset paths back to original paths in the cache.
    Call preload_gather_cache() before processing many files for best performance.
    
    Args:
        file_path: Path to the file
        algorithm: 'md5' or 'sha256' (case-sensitive, use lowercase)
        raw_data_dir: Optional raw_data directory path (only used if cache not preloaded)
        unified_snapshot: Optional UnifiedSnapshot to map paths back to original hashes
        relative_path: Optional relative path (e.g., 'train/coherent/sha256.jpg') for snapshot lookup
        logger: Optional logger instance for warning messages when hash calculation fails
        
    Returns:
        Hash hexdigest, or None on error (file not found, permission denied, I/O error, etc.)
    """
    # For MD5, try to use gather cache first (check global cache directly for speed)
    if algorithm == 'md5':
        # Load cache if not already loaded (fallback for backward compatibility)
        if _gather_cache is None:
            preload_gather_cache(raw_data_dir)
        
        # Try to use unified snapshot to get hash directly (avoids recalculation).
        # Dataset creation copies bytes without changing content, so converted MD5 matches train/test files.
        if unified_snapshot and relative_path:
            # Look up the image record by dataset path
            rp = _posix_rel(relative_path)
            for image_record in unified_snapshot.images.values():
                dataset_info = image_record.get('dataset')
                if dataset_info and _posix_rel(dataset_info.get('path')) == rp:
                    # Found the record — use stored converted MD5 (same bytes as train/test copy).
                    converted_info = image_record.get('converted')
                    if converted_info and isinstance(converted_info, dict):
                        converted_md5 = converted_info.get('md5')
                        if converted_md5:
                            # Return the converted hash directly - no file I/O needed!
                            return converted_md5
        
        # Try cache lookup if available (direct path matching)
        if _gather_cache:
            # Try both absolute and relative path lookups
            cached_hash = _gather_cache.get(str(file_path))
            if cached_hash:
                return cached_hash
            # Also try with resolved absolute path
            try:
                resolved_path = str(file_path.resolve())
                cached_hash = _gather_cache.get(resolved_path)
                if cached_hash:
                    return cached_hash
            except Exception:
                pass
    
    # Calculate hash if not in cache or algorithm is sha256
    try:
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            hash_obj = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read in chunks for memory efficiency
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except FileNotFoundError:
        # File was deleted between glob and hash calculation (race condition)
        if logger:
            logger.warning(f"File not found (deleted?): {file_path}")
        return None
    except PermissionError:
        # Permission denied - shouldn't happen but handle gracefully
        if logger:
            logger.warning(f"Permission denied reading file: {file_path}")
        return None
    except Exception as e:
        # Log other errors for debugging
        if logger:
            logger.warning(f"Error calculating hash for {file_path}: {e}")
        return None


class UnifiedSnapshot:
    """
    Unified snapshot that tracks images through the pipeline (convert, dataset, training).

    Typically persisted as ``snapshot_<run_id>.json`` under ``raw_data`` or ``data``; the
    run ID ties updates together across stages.
    
    Structure: One record per original image, with all pipeline stages nested within.
    This makes analysis easier as no joins are needed.

    Optional top-level ``training_timing`` holds wall-clock seconds for the last
    :class:`~mb.training.trainer.ModelTrainer` run that updated this snapshot (when
    ``update_snapshot`` is enabled).
    """
    
    def __init__(self, run_id: str, raw_data_dir: str, data_dir: Optional[str] = None):
        self.run_id = run_id
        self.raw_data_directory = raw_data_dir
        self.data_directory = data_dir
        self.created_timestamp = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        
        # Single list: one record per original image with all stages nested
        # Keyed by original_hash for fast lookup
        self.images: Dict[str, Dict] = {}  # {original_hash: image_record}
        # Optional: disk space estimates (fingerprints + bytes) from :mod:`mb.space_estimate`
        self.space_estimates: Optional[Dict[str, Any]] = None
        # Optional: wall-clock training summary (set by :class:`~mb.training.trainer.ModelTrainer`)
        self.training_timing: Optional[Dict[str, Any]] = None
        # step → invocation ISO timestamp → list of error lines (see :func:`register_step_error`)
        self.step_errors: Dict[str, Dict[str, List[str]]] = {}

    def add_pre_conversion_image(self, image_path: Path, base_dir: Path) -> bool:
        """Add an image to pre-conversion stage (creates new record)."""
        try:
            # Use raw_data_directory to help locate gather cache
            # Cache is at raw_data/.gather_cache.pkl, so if raw_data_directory is raw_data/coherent,
            # we need the parent. The _load_gather_cache function will try multiple paths.
            raw_data_dir = None
            if self.raw_data_directory:
                raw_data_path = Path(self.raw_data_directory)
                # Try parent first (most common case: raw_data/coherent -> raw_data)
                raw_data_dir = raw_data_path.parent
            image_hash = calculate_file_hash(image_path, algorithm='md5', raw_data_dir=raw_data_dir)
            if image_hash is None:
                return False
            
            relative_path = image_path.relative_to(base_dir)
            
            # Create new image record
            self.images[image_hash] = {
                'original': {
                    'basename': image_path.name,
                    'hash': image_hash,
                    'path': str(relative_path),
                    'format': image_path.suffix.lower()
                },
                'converted': None,
                'dataset': None,
                'training': None
            }
            self.last_updated = datetime.now().isoformat()
            return True
        except Exception:
            return False
    
    def add_post_conversion_image(self, class_name: str, converted_path: str, 
                                   converted_basename: str, converted_md5: str, 
                                   converted_sha256: str, original_info: Optional[Dict] = None) -> None:
        """Add an image after conversion (updates existing record)."""
        # Find the image record by original hash
        original_hash = None
        if original_info:
            original_hash = original_info.get('original_hash')
        
        # If no original_info, try to match by MD5 (file wasn't converted)
        if not original_hash:
            original_hash = converted_md5
        
        # Find or create image record
        if original_hash not in self.images:
            # Create new record if not found (shouldn't happen, but handle gracefully)
            self.images[original_hash] = {
                'original': original_info or {
                    'basename': converted_basename,
                    'hash': original_hash,
                    'path': converted_path,
                    'format': '.jpg'
                },
                'converted': None,
                'dataset': None,
                'training': None
            }
        
        # Update converted stage
        was_converted = (original_hash != converted_md5) if original_hash else None
        self.images[original_hash]['converted'] = {
            'class': class_name,
            'path': converted_path,
            'basename': converted_basename,
            'md5': converted_md5,
            'sha256': converted_sha256,
            'was_converted': was_converted
        }
        self.last_updated = datetime.now().isoformat()
    
    def add_dataset_image(self, class_name: str, converted_path: str, converted_basename: str,
                          converted_md5: str, converted_sha256: str, final_path: str,
                          final_basename: str) -> None:
        """
        Add an image to dataset creation stage (updates existing record).
        
        Args:
            class_name: Class name
            converted_path: Path to converted image (relative to raw_data)
            converted_basename: Basename of converted image
            converted_md5: MD5 hash of converted image
            converted_sha256: SHA256 hash of converted image
            final_path: Final path in dataset (train/class/filename or test/class/filename)
            final_basename: Final basename
        """
        # Try to find image record by MD5 hash first (works if file wasn't converted/resized)
        original_hash = None
        if converted_md5 in self.images:
            original_hash = converted_md5
        else:
            # Try to match by converted path/basename (file was converted/resized)
            for hash_key, img_record in self.images.items():
                converted_info = img_record.get('converted')
                if converted_info and isinstance(converted_info, dict):
                    if (converted_info.get('path') == converted_path or 
                        converted_info.get('basename') == converted_basename):
                        original_hash = hash_key
                        break
        
        # If still not found, try basename match on original
        if not original_hash:
            for hash_key, img_record in self.images.items():
                original = img_record.get('original')
                if original and isinstance(original, dict):
                    orig_basename = original.get('basename', '')
                    if orig_basename.rsplit('.', 1)[0] == converted_basename.rsplit('.', 1)[0]:
                        original_hash = hash_key
                        break
        
        # Create new record if not found (shouldn't happen, but handle gracefully)
        if not original_hash:
            original_hash = converted_md5
            self.images[original_hash] = {
                'original': {
                    'basename': converted_basename,
                    'hash': converted_md5,
                    'path': converted_path,
                    'format': '.jpg'
                },
                'converted': None,
                'dataset': None,
                'training': None
            }
        
        # Update dataset stage
        self.images[original_hash]['dataset'] = {
            'class': class_name,
            'path': final_path,
            'basename': final_basename,
            'sha256': converted_sha256,
            'split': 'train'  # Will be updated if moved to test
        }
        self.last_updated = datetime.now().isoformat()
    
    def update_dataset_split(self, final_path: str, new_split: str) -> bool:
        """Update dataset split for an image."""
        # Find image by dataset final_path
        for image_record in self.images.values():
            if image_record.get('dataset') and image_record['dataset'].get('path') == final_path:
                image_record['dataset']['split'] = new_split
                if new_split == 'test':
                    image_record['dataset']['path'] = image_record['dataset']['path'].replace('train/', 'test/', 1)
                self.last_updated = datetime.now().isoformat()
                return True
        return False
    
    def remove_dataset_image(self, final_path: str) -> bool:
        """Remove an image from dataset creation (sets dataset to None)."""
        # Find image by dataset final_path and remove dataset info
        for image_record in self.images.values():
            if image_record.get('dataset') and image_record['dataset'].get('path') == final_path:
                image_record['dataset'] = None
                self.last_updated = datetime.now().isoformat()
                return True
        return False
    
    def add_training_image(self, split: str, class_name: str, path: str, hash: str, basename: str) -> None:
        """
        Add an image to training stage (updates existing record).
        
        Args:
            split: 'train' or 'test'
            class_name: Class name
            path: Relative path from data directory (e.g., 'train/coherent/sha256.jpg')
            hash: MD5 hash of the image
            basename: Basename of the image file
        """
        path_key = _posix_rel(path)
        # Find image record by matching dataset path (most reliable)
        found = False
        for image_record in self.images.values():
            dataset_info = image_record.get('dataset')
            if dataset_info and _posix_rel(dataset_info.get('path')) == path_key:
                # Get original and converted info from the record itself
                original = image_record.get('original', {})
                converted = image_record.get('converted', {})
                
                image_record['training'] = {
                    'split': split,
                    'class': class_name,
                    'path': path_key,
                    'hash': hash,
                    'basename': basename,
                    'original_hash': original.get('hash'),
                    'original_path': original.get('path'),
                    'original_format': original.get('format'),
                    'was_converted': converted.get('was_converted') if converted else None
                }
                found = True
                break
        
        # If not found by path, try to match by hash (fallback)
        if not found and hash:
            for image_record in self.images.values():
                original = image_record.get('original', {})
                if original.get('hash') == hash:
                    converted = image_record.get('converted', {})
                    image_record['training'] = {
                        'split': split,
                        'class': class_name,
                        'path': path_key,
                        'hash': hash,
                        'basename': basename,
                        'original_hash': original.get('hash'),
                        'original_path': original.get('path'),
                        'original_format': original.get('format'),
                        'was_converted': converted.get('was_converted') if converted else None
                    }
                    found = True
                    break
        
        if found:
            self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        # Convert images dict to list for JSON serialization
        images_list = list(self.images.values())
        
        # Calculate summary statistics
        total_images = len(images_list)
        converted_count = sum(1 for img in images_list if img.get('converted') is not None)
        dataset_count = sum(1 for img in images_list if img.get('dataset') is not None)
        training_count = sum(1 for img in images_list if img.get('training') is not None)
        train_count = sum(1 for img in images_list 
                         if img.get('training') and img['training'].get('split') == 'train')
        test_count = sum(1 for img in images_list 
                        if img.get('training') and img['training'].get('split') == 'test')
        dataset_train_count = sum(1 for img in images_list 
                                 if img.get('dataset') and img['dataset'].get('split') == 'train')
        dataset_test_count = sum(1 for img in images_list 
                                if img.get('dataset') and img['dataset'].get('split') == 'test')
        
        out: Dict[str, Any] = {
            'run_id': self.run_id,
            'created_timestamp': self.created_timestamp,
            'last_updated': self.last_updated,
            'raw_data_directory': self.raw_data_directory,
            'data_directory': self.data_directory,
            'images': images_list,
            'summary': {
                'total_images': total_images,
                'converted_count': converted_count,
                'dataset_count': dataset_count,
                'dataset_train_count': dataset_train_count,
                'dataset_test_count': dataset_test_count,
                'training_count': training_count,
                'training_train_count': train_count,
                'training_test_count': test_count
            }
        }
        if self.space_estimates is not None:
            out['space_estimates'] = self.space_estimates
        if self.training_timing is not None:
            out['training_timing'] = self.training_timing
        if self.step_errors:
            out['step_errors'] = self.step_errors
        return out
    
    def save(self, output_path: Path) -> bool:
        """Save unified snapshot to JSON file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    @classmethod
    def load(cls, snapshot_path: Path) -> Optional['UnifiedSnapshot']:
        """Load unified snapshot from JSON file."""
        logger.info(f"Loading unified snapshot from: {snapshot_path}")
        if not snapshot_path.exists():
            return None
        
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            snapshot = cls(
                run_id=data['run_id'],
                raw_data_dir=data['raw_data_directory'],
                data_dir=data.get('data_directory')
            )
            snapshot.created_timestamp = data.get('created_timestamp', snapshot.created_timestamp)
            snapshot.last_updated = data.get('last_updated', snapshot.last_updated)
            
            # Convert images list back to dict keyed by original hash
            images_list = data.get('images', [])
            for img_record in images_list:
                original_hash = img_record.get('original', {}).get('hash')
                if original_hash:
                    snapshot.images[original_hash] = img_record

            se = data.get('space_estimates')
            snapshot.space_estimates = se if isinstance(se, dict) else None
            tt = data.get('training_timing')
            snapshot.training_timing = tt if isinstance(tt, dict) else None

            se = data.get("step_errors")
            if isinstance(se, dict):
                snapshot.step_errors = _coerce_loaded_step_errors(se)
            else:
                snapshot.step_errors = {}

            return snapshot
        except Exception:
            return None


def find_latest_unified_snapshot_path(search_paths: List[Path]) -> Optional[Path]:
    """
    Return the path to the **newest** ``snapshot_*.json`` (by file mtime) under any of
    *search_paths*, or ``None`` if none exist.
    """
    logger.info("Finding latest unified snapshot path…")
    candidates: List[Path] = []
    for d in search_paths:
        try:
            p = Path(d)
        except TypeError:
            continue
        if p.is_dir():
            try:
                candidates.extend(p.glob("snapshot_*.json"))
            except OSError:
                continue
    if not candidates:
        return None

    def _mtime_key(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return -1.0

    try:
        return max(candidates, key=_mtime_key)
    except ValueError:
        return None


def format_latest_unified_snapshot_summary(search_paths: List[Path]) -> str:
    """
    Short multi-line summary for UI (absolute path, run id, image count, last updated).

    Uses :func:`find_latest_unified_snapshot_path`; reads JSON on success.
    """
    path = find_latest_unified_snapshot_path(search_paths)
    if path is None or not path.exists():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return str(path.resolve())
    rid = data.get("run_id", "?")
    lu = str(data.get("last_updated") or "")[:22]
    summary = data.get("summary") or {}
    n = summary.get("total_images")
    if n is None and isinstance(data.get("images"), list):
        n = len(data["images"])
    lines = [
        str(path.resolve()),
        f"run_id: {rid}",
        f"last_updated: {lu}",
        f"images: {n if n is not None else '?'}",
    ]
    return "\n".join(lines)


def find_unified_snapshot(search_paths: List[Path], run_id: Optional[str] = None, logger=None) -> Optional[UnifiedSnapshot]:
    """
    Find and load unified snapshot by run ID or latest.
    
    Args:
        search_paths: List of directories to search
        run_id: Optional run ID to find specific snapshot
        logger: Optional logger for messages
        
    Returns:
        UnifiedSnapshot instance, or None if not found
    """
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        if run_id:
            # Look for specific run ID
            snapshot_path = search_path / f"snapshot_{run_id}.json"
            if snapshot_path.exists():
                snapshot = UnifiedSnapshot.load(snapshot_path)
                if snapshot:
                    if logger:
                        logger.info(f"Loaded unified snapshot with run_id {run_id} from: {snapshot_path}")
                    return snapshot
        else:
            # Look for latest snapshot
            snapshot_files = sorted(search_path.glob("snapshot_*.json"))
            if snapshot_files:
                # Try to load the most recent one
                for snapshot_path in reversed(snapshot_files):
                    snapshot = UnifiedSnapshot.load(snapshot_path)
                    if snapshot:
                        if logger:
                            logger.info(f"Loaded unified snapshot with run_id {snapshot.run_id} from: {snapshot_path}")
                        return snapshot
    
    if logger:
        logger.info("No unified snapshot found")
    return None


def save_unified_snapshot(snapshot: UnifiedSnapshot, output_dir: Path, logger=None) -> Optional[Path]:
    """
    Save unified snapshot with run ID filename.
    
    Args:
        snapshot: UnifiedSnapshot instance to save
        output_dir: Directory to save snapshot
        logger: Optional logger for messages
        
    Returns:
        Path where snapshot was saved, or None on failure
    """
    snapshot_path = output_dir / f"snapshot_{snapshot.run_id}.json"
    if snapshot.save(snapshot_path):
        if logger:
            logger.info(f"Unified snapshot saved to: {snapshot_path}")
        return snapshot_path
    return None
