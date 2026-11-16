"""
Model Builder - A unified CLI application for building machine learning models.

This package provides a framework-agnostic interface for training models,
with support for PyTorch, Keras/TensorFlow, and extensible architecture for
additional frameworks and model types.
"""

from mb.utils.version import app_version as _app_version

try:
    # The repo-root VERSION file, bundled in a build; the only place the version is kept.
    __version__ = _app_version()
except OSError:
    __version__ = "unknown"
