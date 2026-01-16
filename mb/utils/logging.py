"""
Centralized logging configuration for Model Builder.

This module provides consistent logging setup across all components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    script_name: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for a script.
    
    Args:
        log_file: Name of the log file (optional). If None, uses script name + '.log'
                  If a relative path is provided, it will be placed in the logs/ directory.
                  If an absolute path is provided, it will be used as-is.
        log_level: Logging level (default: INFO)
        script_name: Name of the script for the logger (optional)
    
    Returns:
        Configured logger instance
    """
    # Determine log file name
    if log_file is None and script_name is None:
        # Try to get script name from sys.argv[0]
        script_name = Path(sys.argv[0]).stem if sys.argv else "mb"
        log_file = f"{script_name}.log"
    elif log_file is None:
        log_file = f"{script_name}.log"
    
    # Ensure logs directory exists and handle log file path
    log_file_path = Path(log_file)
    
    # If log_file is not an absolute path, place it in logs/ directory
    if not log_file_path.is_absolute():
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        # Preserve subdirectories if present, otherwise just use filename
        if log_file_path.parent != Path("."):
            log_file_path = logs_dir / log_file_path
        else:
            log_file_path = logs_dir / log_file_path.name
    
    # Ensure parent directory exists (in case log_file_path has subdirectories)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger_name = script_name if script_name else "mb"
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set log level
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Logger instance
    """
    if name is None:
        # Try to get script name from sys.argv[0]
        name = Path(sys.argv[0]).stem if sys.argv else "mb"
    
    return logging.getLogger(name)


def log_startup_info(logger: logging.Logger, script_description: Optional[str] = None) -> None:
    """
    Log startup information for a script.
    
    Args:
        logger: Logger instance
        script_description: Description of what the script does
    """
    logger.info("=" * 80)
    logger.info("SCRIPT STARTUP")
    logger.info("=" * 80)
    
    if script_description:
        logger.info(f"Script: {script_description}")
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Log file: {logger.handlers[0].baseFilename if logger.handlers else 'Not configured'}")


def log_completion_info(logger: logging.Logger, success: bool = True, message: Optional[str] = None) -> None:
    """
    Log completion information for a script.
    
    Args:
        logger: Logger instance
        success: Whether the script completed successfully
        message: Optional completion message
    """
    logger.info("=" * 80)
    logger.info("SCRIPT COMPLETION")
    logger.info("=" * 80)
    
    if success:
        logger.info("Script completed successfully")
    else:
        logger.error("Script completed with errors")
    
    if message:
        logger.info(f"Message: {message}")


def quick_setup(script_name: Optional[str] = None) -> logging.Logger:
    """
    Quick setup for logging with default settings.
    
    Args:
        script_name: Name of the script (optional)
    
    Returns:
        Configured logger instance
    """
    return setup_logging(script_name=script_name)
