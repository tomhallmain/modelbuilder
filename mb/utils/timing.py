#!/usr/bin/env python3
"""
Timing utilities for tracking execution times in Model Builder.
Provides comprehensive time tracking with JSON storage for analysis.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging


class TimingTracker:
    """
    Comprehensive timing tracker for training and other long-running processes.
    """
    
    def __init__(self, log_dir: str = "logs", timing_dir: str = "timing_data"):
        """
        Initialize the timing tracker.
        
        Args:
            log_dir: Directory for log files
            timing_dir: Directory for timing data files
        """
        self.log_dir = Path(log_dir)
        self.timing_dir = Path(timing_dir)
        
        # Create directories if they don't exist
        self.log_dir.mkdir(exist_ok=True)
        self.timing_dir.mkdir(exist_ok=True)
        
        # Timing data
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
        self.phases = {}
        self.current_phase = None
        self.phase_start_time = None
        
        # Generate unique session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def start_session(self, session_name: Optional[str] = None) -> None:
        """
        Start a new timing session.
        
        Args:
            session_name: Optional name for the session
        """
        self.start_time = time.time()
        self.session_name = session_name or f"session_{self.session_id}"
        
        # Log the start
        logger = logging.getLogger(__name__)
        logger.info(f"Timing session started: {self.session_name}")
        logger.info(f"Session ID: {self.session_id}")
        
    def end_session(self) -> Dict[str, Any]:
        """
        End the current timing session and return timing data.
        
        Returns:
            Dictionary containing all timing information
        """
        self.end_time = time.time()
        
        # Calculate total duration
        total_duration = self.end_time - self.start_time
        
        # Prepare timing data
        timing_data = {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "total_duration_seconds": total_duration,
            "total_duration_formatted": str(timedelta(seconds=int(total_duration))),
            "checkpoints": self.checkpoints,
            "phases": self.phases
        }
        
        # Log the end
        logger = logging.getLogger(__name__)
        logger.info(f"Timing session ended: {self.session_name}")
        logger.info(f"Total duration: {timing_data['total_duration_formatted']}")
        
        return timing_data
    
    def add_checkpoint(self, name: str, description: Optional[str] = None) -> None:
        """
        Add a checkpoint with current timestamp.
        
        Args:
            name: Checkpoint name
            description: Optional description
        """
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        self.checkpoints[name] = {
            "timestamp": datetime.fromtimestamp(current_time).isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_formatted": str(timedelta(seconds=int(elapsed))),
            "description": description
        }
        
        # Log the checkpoint
        logger = logging.getLogger(__name__)
        logger.info(f"Checkpoint '{name}': {self.checkpoints[name]['elapsed_formatted']} elapsed")
        if description:
            logger.info(f"  Description: {description}")
    
    def start_phase(self, phase_name: str, description: Optional[str] = None) -> None:
        """
        Start a new phase of execution.
        
        Args:
            phase_name: Name of the phase
            description: Optional description
        """
        if self.current_phase:
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        
        self.phases[phase_name] = {
            "start_time": datetime.fromtimestamp(self.phase_start_time).isoformat(),
            "description": description,
            "duration_seconds": None,
            "duration_formatted": None
        }
        
        # Log the phase start
        logger = logging.getLogger(__name__)
        logger.info(f"Phase started: {phase_name}")
        if description:
            logger.info(f"  Description: {description}")
    
    def end_phase(self) -> Optional[Dict[str, Any]]:
        """
        End the current phase and return its timing data.
        
        Returns:
            Phase timing data or None if no phase was active
        """
        if not self.current_phase or not self.phase_start_time:
            return None
        
        end_time = time.time()
        duration = end_time - self.phase_start_time
        
        phase_data = self.phases[self.current_phase]
        phase_data.update({
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "duration_formatted": str(timedelta(seconds=int(duration)))
        })
        
        # Log the phase end
        logger = logging.getLogger(__name__)
        logger.info(f"Phase ended: {self.current_phase}")
        logger.info(f"  Duration: {phase_data['duration_formatted']}")
        
        self.current_phase = None
        self.phase_start_time = None
        
        return phase_data
    
    def save_timing_data(self, timing_data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Save timing data to a JSON file.
        
        Args:
            timing_data: Timing data dictionary
            filename: Optional filename (defaults to session-based name)
        
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"timing_{self.session_id}.json"
        
        filepath = self.timing_dir / filename
        
        # Add metadata
        timing_data["metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(filepath, 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        # Log the save
        logger = logging.getLogger(__name__)
        logger.info(f"Timing data saved to: {filepath}")
        
        return filepath
    
    def get_elapsed_time(self) -> str:
        """
        Get the elapsed time since session start as a formatted string.
        
        Returns:
            Formatted elapsed time string
        """
        if not self.start_time:
            return "0:00:00"
        
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    return str(timedelta(seconds=int(seconds)))


def log_timing_summary(logger: logging.Logger, timing_data: Dict[str, Any]) -> None:
    """
    Log a summary of timing data.
    
    Args:
        logger: Logger instance
        timing_data: Timing data dictionary
    """
    logger.info("=" * 60)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Session: {timing_data['session_name']}")
    logger.info(f"Total duration: {timing_data['total_duration_formatted']}")
    
    if timing_data['phases']:
        logger.info("\nPhases:")
        for phase_name, phase_data in timing_data['phases'].items():
            if phase_data.get('duration_formatted'):
                logger.info(f"  {phase_name}: {phase_data['duration_formatted']}")
                if phase_data.get('description'):
                    logger.info(f"    {phase_data['description']}")
    
    if timing_data['checkpoints']:
        logger.info("\nCheckpoints:")
        for checkpoint_name, checkpoint_data in timing_data['checkpoints'].items():
            logger.info(f"  {checkpoint_name}: {checkpoint_data['elapsed_formatted']}")
            if checkpoint_data.get('description'):
                logger.info(f"    {checkpoint_data['description']}")
    
    logger.info("=" * 60)
