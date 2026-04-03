from abc import ABC
import json
import os
import shutil
import threading
from typing import Any, Dict, List

from utils.constants import AppInfo
from utils.encryptor import encrypt_data_to_file, decrypt_data_from_file
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class InflationMonitor(ABC):
    """
    Abstract base class for cache systems that need to monitor list inflation.
    Concrete classes must implement get_cache_dict() and optionally get_meta() to provide the cache dictionary.
    """
    
    # Configuration
    INFLATION_ENABLED_KEY = "__monitor_list_inflation"
    INFLATION_MIN_LIST_SIZE = 5
    INFLATION_GROWTH_THRESHOLD = 2.0
    INFLATION_MAX_LIST_SIZE = 1000
    
    def __init__(self):
        self._monitor_inflation = False
        self._initial_list_sizes = {}
        self._suspected_keys = set()
        self._check_monitoring_enabled()
    
    def get_cache_dict(self, scope_key: str = "info") -> Dict:
        """Get the cache dictionary to monitor (required by InflationMonitor)."""
        raise NotImplementedError("get_cache_dict is not implemented in the base class")
    
    def get_meta(self, key: str, default_val: Any = None) -> Any:
        """
        Optional method to get meta value. If not implemented, monitoring will only check environment variable.
        Override this if your cache supports meta storage.
        """
        return default_val
    
    def _check_monitoring_enabled(self, scope_key: str = "info"):
        """Check if monitoring should be enabled via environment variable or meta flag."""
        if os.environ.get("MONITOR_CACHE_INFLATION", "0") == "1":
            self.enable_inflation_monitoring(True, scope_key)
            return

        try:
            monitor_flag = self.get_meta(self.INFLATION_ENABLED_KEY, False)
            if monitor_flag:
                self.enable_inflation_monitoring(True, scope_key)
        except (AttributeError, NotImplementedError):
            # get_meta not implemented or not available yet, skip
            pass
    
    def enable_inflation_monitoring(self, enable: bool = True, scope_key: str = "info"):
        """Enable or disable inflation monitoring."""
        if enable:
            self._monitor_inflation = True
            self._record_initial_list_sizes(scope_key)
            logger.info(f"Inflation monitoring enabled. Tracking {len(self._initial_list_sizes)} lists.")
        else:
            self._monitor_inflation = False
            self._initial_list_sizes = {}
            self._suspected_keys = set()
            logger.info("Inflation monitoring disabled.")
    
    def _record_initial_list_sizes(self, scope_key: str):
        """Record initial list sizes from cache with debug logging."""
        cache_dict = self.get_cache_dict(scope_key)
        self._initial_list_sizes = {}
        
        def _traverse_and_record(data: Any, path: str):
            """Recursively find and record list sizes in dictionaries."""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, list):
                        self._initial_list_sizes[current_path] = len(value)
                        logger.debug(f"Tracking list: {current_path} = {len(value)} items")
                        
                        # Check for immediate issues with large lists
                        if len(value) > self.INFLATION_MAX_LIST_SIZE:
                            logger.warning(f"Large initial list at {current_path}: {len(value)} items")
                    elif isinstance(value, dict):
                        _traverse_and_record(value, current_path)
            elif isinstance(data, list):
                self._initial_list_sizes[path] = len(data)
                logger.debug(f"Tracking list: {path} = {len(data)} items")
                if len(data) > self.INFLATION_MAX_LIST_SIZE:
                    logger.warning(f"Large initial list at {path}: {len(data)} items")
        
        if cache_dict:
            _traverse_and_record(cache_dict, path=scope_key)
    
    def _collect_list_sizes(self, data: Any, path: str = "") -> Dict[str, int]:
        """Collect sizes of all lists in a data structure."""
        sizes = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    sizes.update(self._collect_list_sizes(value, current_path))
        elif isinstance(data, list):
            sizes[path] = len(data)
        
        return sizes
    
    def check_for_inflation(self, scope_key: str = "info") -> List[str]:
        """
        Check for list inflation and return list of suspected keys with detailed logging.
        
        Returns:
            List of key paths suspected of inflation
        """
        if not self._monitor_inflation:
            return []
        
        cache_dict = self.get_cache_dict(scope_key)
        current_sizes = self._collect_list_sizes(cache_dict, scope_key)
        
        suspected = []
        inflation_detected = False
        
        for list_path, initial_size in self._initial_list_sizes.items():
            if list_path in current_sizes:
                current_size = current_sizes[list_path]
                
                # Skip very small lists
                if initial_size < self.INFLATION_MIN_LIST_SIZE:
                    continue
                
                # Check for doubling or significant growth
                if current_size >= initial_size * self.INFLATION_GROWTH_THRESHOLD:
                    inflation_detected = True
                    suspected.append(list_path)
                    
                    if list_path not in self._suspected_keys:
                        logger.warning(
                            f"⚠️ LIST INFLATION DETECTED: {list_path}\n"
                            f"   Initial size: {initial_size}\n"
                            f"   Current size: {current_size}\n"
                            f"   Growth factor: {current_size/initial_size:.1f}x\n"
                            f"   This may indicate duplicate entries or append-instead-of-update."
                        )
                        
                        # Try to identify if it's likely a duplication issue
                        if current_size % initial_size == 0:
                            multiples = current_size // initial_size
                            logger.info(f"   Note: Current size is exactly {multiples} times initial size.")
        
        if not inflation_detected and self._initial_list_sizes:
            logger.debug(f"No list inflation detected in '{scope_key}' dictionary.")
        
        self._suspected_keys = set(suspected)
        return suspected
    
    def get_suspected_inflation_keys(self, scope_key: str = "info") -> List[str]:
        """
        Get list of keys suspected of inflation with formatted details.
        
        Returns:
            List of formatted strings describing suspected inflation
        """
        if not self._monitor_inflation:
            return []
        
        suspected = []
        cache_dict = self.get_cache_dict(scope_key)
        current_sizes = self._collect_list_sizes(cache_dict, scope_key)
        
        for list_path, initial_size in self._initial_list_sizes.items():
            if list_path in current_sizes:
                current_size = current_sizes[list_path]
                if (initial_size >= self.INFLATION_MIN_LIST_SIZE and 
                    current_size >= initial_size * self.INFLATION_GROWTH_THRESHOLD):
                    suspected.append(f"{list_path} ({initial_size} → {current_size}, {current_size/initial_size:.1f}x)")
        
        return suspected
    
    def get_inflation_report(self) -> Dict[str, Any]:
        """Get inflation monitoring report."""
        return {
            "enabled": self._monitor_inflation,
            "tracked_lists": len(self._initial_list_sizes),
            "suspected_keys": list(self._suspected_keys)
        }


class AppInfoCache(InflationMonitor):
    CACHE_LOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_info_cache.enc")
    JSON_LOC = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "app_info_cache.json")
    META_INFO_KEY = "info"
    NUM_BACKUPS = 4

    def __init__(self):
        super().__init__()
        self._lock = threading.RLock()
        self._cache = {self.META_INFO_KEY: {}}
        self.load()
        self.validate()

    def _normalize_loaded_cache(self, data: Any) -> None:
        """Drop legacy keys (e.g. per-directory map) and ensure ``info`` exists."""
        if not isinstance(data, dict):
            self._cache = {self.META_INFO_KEY: {}}
            return
        data.pop("directories", None)
        if self.META_INFO_KEY not in data or not isinstance(data[self.META_INFO_KEY], dict):
            data[self.META_INFO_KEY] = {}
        self._cache = data

    def get_cache_dict(self, scope_key: str = "info") -> Dict:
        """Get the cache dictionary to monitor (required by InflationMonitor)."""
        with self._lock:
            if scope_key == "info":
                return self._cache.get(self.META_INFO_KEY, {})
            return self._cache.get(scope_key, {})

    def store(self):
        """Persist cache to encrypted file. Returns True on success, False if encrypted store failed but JSON fallback succeeded. Raises on encoding or JSON fallback failure."""
        with self._lock:
            if self._monitor_inflation:
                self.check_for_inflation()

            try:
                cache_data = json.dumps(self._cache).encode('utf-8')
            except Exception as e:
                raise Exception(f"Error compiling application cache: {e}")

            try:
                encrypt_data_to_file(
                    cache_data,
                    AppInfo.SERVICE_NAME,
                    AppInfo.APP_IDENTIFIER,
                    AppInfoCache.CACHE_LOC,
                )
                return True
            except Exception as e:
                logger.error(f"Error encrypting cache: {e}")

            try:
                with open(AppInfoCache.JSON_LOC, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f)
                return False  # Encryption failed, but JSON fallback succeeded
            except Exception as e:
                raise Exception(f"Error storing application cache: {e}") from e

    def _try_load_cache_from_file(self, path):
        encrypted_data = decrypt_data_from_file(path, AppInfo.SERVICE_NAME, AppInfo.APP_IDENTIFIER)
        return json.loads(encrypted_data.decode("utf-8"))

    def load(self):
        with self._lock:
            try:
                if os.path.exists(AppInfoCache.JSON_LOC):
                    logger.info("Detected JSON-format application cache, will attempt migration to encrypted store")
                    with open(AppInfoCache.JSON_LOC, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    self._normalize_loaded_cache(raw)
                    if self.store():
                        logger.info(f"Migrated application cache from {AppInfoCache.JSON_LOC} to encrypted store")
                        os.remove(AppInfoCache.JSON_LOC)
                    else:
                        logger.warning("Encrypted store of application cache failed; keeping JSON cache file")
                    return

                # Try encrypted cache and backups in order
                cache_paths = [self.CACHE_LOC] + self._get_backup_paths()
                any_exist = any(os.path.exists(path) for path in cache_paths)
                if not any_exist:
                    logger.info(f"No cache file found at {self.CACHE_LOC}, creating new cache")
                    return

                for path in cache_paths:
                    if os.path.exists(path):
                        try:
                            raw = self._try_load_cache_from_file(path)
                            self._normalize_loaded_cache(raw)
                            if path == self.CACHE_LOC:
                                message = f"Loaded cache from {self.CACHE_LOC}"
                                rotated_count = self._rotate_backups()
                                if rotated_count > 0:
                                    message += f", rotated {rotated_count} backups"
                                logger.info(message)
                            else:
                                logger.warning(f"Loaded cache from backup: {path}")
                            return
                        except Exception as e:
                            logger.error(f"Failed to load cache from {path}: {e}")
                            continue
                # If we get here, all attempts failed (but at least one file existed)
                raise Exception(f"Failed to load cache from all locations: {cache_paths}")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                raise e

    def validate(self) -> None:
        """No-op (directory validation removed; cache holds app meta only)."""
        return

    def set(self, key, value):
        with self._lock:
            if self.META_INFO_KEY not in self._cache:
                self._cache[self.META_INFO_KEY] = {}
            self._cache[self.META_INFO_KEY][key] = value

    def get(self, key, default_val=None):
        with self._lock:
            info = self._cache.get(self.META_INFO_KEY, {})
            if key not in info:
                return default_val
            return info[key]

    def get_meta(self, key: str, default_val: Any = None) -> Any:
        """Same as :meth:`get`; kept for :class:`InflationMonitor` hook."""
        return self.get(key, default_val)

    def export_as_json(self, json_path=None):
        """Export the current cache as a JSON file (not encrypted)."""
        if json_path is None:
            json_path = AppInfoCache.JSON_LOC
        with self._lock:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        return json_path

    def _get_backup_paths(self):
        """Get list of backup file paths in order of preference"""
        backup_paths = []
        for i in range(1, self.NUM_BACKUPS + 1):
            index = "" if i == 1 else f"{i}"
            path = f"{self.CACHE_LOC}.bak{index}"
            backup_paths.append(path)
        return backup_paths

    def _rotate_backups(self):
        """Rotate backup files: move each backup to the next position, oldest gets overwritten"""
        backup_paths = self._get_backup_paths()
        rotated_count = 0
        
        # Remove the oldest backup if it exists
        if os.path.exists(backup_paths[-1]):
            os.remove(backup_paths[-1])
        
        # Shift backups: move each backup to the next position
        for i in range(len(backup_paths) - 1, 0, -1):
            if os.path.exists(backup_paths[i - 1]):
                shutil.copy2(backup_paths[i - 1], backup_paths[i])
                rotated_count += 1
        
        # Copy main cache to first backup position
        shutil.copy2(self.CACHE_LOC, backup_paths[0])
        
        return rotated_count


app_info_cache = AppInfoCache()
# app_info_cache.enable_inflation_monitoring(True)
