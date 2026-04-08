import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import FrozenSet, List, Optional

from mb.utils.custom_formatter import CustomFormatter

# When MODELBUILDER_TEST_APP_DATA is set (pytest), these modules log at WARNING so
# routine INFO from config reload (e.g. "Loaded application configuration…") does
# not flood pytest's stderr. Other ``modelbuilder.*`` loggers stay at DEBUG.
# Must match the ``module_name`` passed to :func:`get_logger` (``modelbuilder.{name}``).
_TEST_QUIET_LOG_MODULES = frozenset(
    {
        "config",
        "mb.pipeline_config",
        "cache_controller",
        "utils.app_info_cache",
    }
)


def _test_quiet_logger_full_names() -> FrozenSet[str]:
    """``logging`` names (``modelbuilder.*``) for modules quieted under pytest."""
    return frozenset(f"modelbuilder.{m}" for m in _TEST_QUIET_LOG_MODULES)


def get_log_directory() -> Path:
    """
    Directory for ``modelbuilder_*.log`` files — the same path used when
    :func:`get_logger` attaches its file handler. Ensures the directory exists.

    When ``MODELBUILDER_TEST_APP_DATA`` is set (pytest), logs go under that
    directory so tests do not write to the real app-data folder.
    """
    test_root = os.environ.get("MODELBUILDER_TEST_APP_DATA")
    if test_root:
        log_dir = Path(test_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    appdata_dir: str = os.getenv("APPDATA") if sys.platform == "win32" else os.path.expanduser("~/.local/share")
    log_dir: Path = Path(appdata_dir) / "ModelBuilder" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _cleanup_old_logs(log_dir: Path, logger: logging.Logger) -> None:
    """
    Clean up log files that are older than 30 days if there are more than 10 log files.
    
    Args:
        log_dir: Path object pointing to the directory containing log files
        logger: Logger instance to use for logging cleanup operations
    """
    try:
        log_files: List[Path] = list(log_dir.glob('modelbuilder_*.log'))
        if len(log_files) <= 10:
            return

        current_time: datetime = datetime.now()
        cutoff_date: datetime = current_time - timedelta(days=30)
        
        for log_file in log_files:
            try:
                # Extract date from filename (format: modelbuilder_YYYY-MM-DD.log)
                date_str: str = log_file.stem.split('_')[-1]
                file_date: datetime = datetime.strptime(date_str, '%Y-%m-%d')
            except (ValueError, IndexError):
                # If filename doesn't contain a valid date, use the file's last modified date
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if file_date < cutoff_date:
                log_file.unlink()
                logger.debug(f"Deleted old log file: {log_file}")
    except Exception as e:
        logger.error(f"Error cleaning up old log files: {e}")

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        module_name: The name of the module requesting the logger
        
    Returns:
        A configured logger instance for the module
    """
    # Create logger with module name
    logger: logging.Logger = logging.getLogger(f"modelbuilder.{module_name}")
    test_app_data = os.environ.get("MODELBUILDER_TEST_APP_DATA")
    if test_app_data and module_name in _TEST_QUIET_LOG_MODULES:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # If handlers are already set up, return the logger (may have been created before
    # MODELBUILDER_TEST_APP_DATA was set — e.g. pytest loads tests/ui/conftest.py first).
    if logger.handlers:
        if test_app_data and module_name in _TEST_QUIET_LOG_MODULES:
            logger.setLevel(logging.WARNING)
            for h in logger.handlers:
                h.setLevel(logging.WARNING)
        return logger

    # create console handler with a higher log level
    ch: logging.StreamHandler = logging.StreamHandler()
    ch.setLevel(logging.WARNING if test_app_data and module_name in _TEST_QUIET_LOG_MODULES else logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    log_dir = get_log_directory()

    # Clean up old logs before creating new one
    _cleanup_old_logs(log_dir, logger)

    date_str: str = datetime.now().strftime("%Y-%m-%d")
    log_file: Path = log_dir / f'modelbuilder_{date_str}.log'

    # Add file handler
    fh: logging.FileHandler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.WARNING if test_app_data and module_name in _TEST_QUIET_LOG_MODULES else logging.DEBUG)
    fh.setFormatter(CustomFormatter())
    logger.addHandler(fh)

    return logger


def _sanitize_cli_logger_suffix(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    script_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for a CLI command or subcommand.

    ``log_file`` is accepted for API compatibility; file output always uses the
    shared daily log under ``%APPDATA%/ModelBuilder/logs`` (Windows) or
    ``~/.local/share/ModelBuilder/logs`` (Unix).
    """
    if script_name is None:
        script_name = Path(sys.argv[0]).stem if sys.argv else "mb"
    _ = log_file  # reserved for future per-job log paths
    safe = _sanitize_cli_logger_suffix(script_name)
    logger = get_logger(f"mb.{safe}")
    logger.setLevel(log_level)
    for h in logger.handlers:
        h.setLevel(log_level)
    return logger


def _first_log_file_path(logger: logging.Logger) -> str:
    for h in logger.handlers:
        p = getattr(h, "baseFilename", None)
        if p:
            return str(p)
    return "Not configured"


def log_startup_info(logger: logging.Logger, job_description: Optional[str] = None) -> None:
    logger.info("=" * 80)
    logger.info("JOB STARTUP")
    logger.info("=" * 80)
    if job_description:
        logger.info(f"Job: {job_description}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Log file: {_first_log_file_path(logger)}")


def log_completion_info(
    logger: logging.Logger,
    success: bool = True,
    message: Optional[str] = None,
) -> None:
    logger.info("=" * 80)
    logger.info("JOB COMPLETION")
    logger.info("=" * 80)
    if success:
        logger.info("Job completed successfully")
    else:
        logger.error("Job completed with errors")
    if message:
        logger.info(f"Message: {message}")


def quick_setup(script_name: Optional[str] = None) -> logging.Logger:
    return setup_logging(script_name=script_name)


def apply_application_log_settings() -> None:
    """
    Align root and ``modelbuilder.*`` / ``mb.*`` loggers with
    :class:`utils.config.ApplicationConfig` (``debug``, ``debug2``, ``log_level``).

    When ``debug`` or ``debug2`` is true, the effective level is DEBUG; otherwise
    ``log_level`` names a :mod:`logging` level (e.g. ``info``, ``warning``).
    """
    level = logging.INFO
    try:
        from utils.config import get_application_config

        app = get_application_config().app
        if app.debug or app.debug2:
            level = logging.DEBUG
        else:
            name = (app.log_level or "info").upper()
            cand = getattr(logging, name, None)
            level = cand if isinstance(cand, int) else logging.INFO
    except Exception:
        level = logging.INFO

    test_app_data = os.environ.get("MODELBUILDER_TEST_APP_DATA")
    quiet_names = _test_quiet_logger_full_names() if test_app_data else frozenset()

    def _apply_to_logger(log: logging.Logger, name: str) -> None:
        eff = level
        if name in quiet_names:
            eff = logging.WARNING
        log.setLevel(eff)
        for handler in log.handlers:
            handler.setLevel(eff)

    for name in list(logging.Logger.manager.loggerDict.keys()):
        if not isinstance(name, str):
            continue
        if not (name.startswith("modelbuilder") or name.startswith("mb")):
            continue
        candidate = logging.Logger.manager.loggerDict[name]
        if not isinstance(candidate, logging.Logger):
            continue
        _apply_to_logger(candidate, name)

    _apply_to_logger(logging.getLogger("root"), "root")


def set_logger_level(debug: bool) -> None:
    """
    Set the logger level to DEBUG if debug is True, otherwise set it to INFO.
    This updates all existing loggers in the application hierarchy and their handlers.
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Update all existing loggers in the application hierarchy
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('modelbuilder'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            # Also update all handlers for this logger
            for handler in logger.handlers:
                handler.setLevel(level)
    
    # Also update the root logger for backward compatibility
    root_logger = get_logger("root")
    root_logger.setLevel(level)
    # Update all handlers for root logger
    for handler in root_logger.handlers:
        handler.setLevel(level)

# Initialize root logger for backward compatibility
root_logger: logging.Logger = get_logger("root")
