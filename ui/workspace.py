"""
Workspace state: project root, optional config path, persisted with QSettings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings


@dataclass
class Workspace:
    """Paths the GUI treats as the current project scope."""

    root: Optional[Path] = None
    config_path: Optional[Path] = None

    @classmethod
    def load(cls, settings: QSettings) -> "Workspace":
        root_s = settings.value("workspace/root", "")
        cfg_s = settings.value("workspace/config", "")
        root = Path(root_s) if root_s else None
        cfg = Path(cfg_s) if cfg_s else None
        return cls(root=root, config_path=cfg)

    def save(self, settings: QSettings) -> None:
        settings.setValue("workspace/root", str(self.root) if self.root else "")
        settings.setValue("workspace/config", str(self.config_path) if self.config_path else "")


def effective_pipeline_config_path(ws: Workspace) -> Optional[Path]:
    """
    Pipeline YAML for ``reload_pipeline_config``: ``configs/pipeline.yaml`` under the
    workspace root, else the explicit menu config file, else legacy ``default.yaml``.
    Matches :meth:`MainWindow._effective_pipeline_config_path`.
    """
    if ws.root:
        pipe = ws.root / "configs" / "pipeline.yaml"
        if pipe.is_file():
            return pipe
    if ws.config_path is not None and ws.config_path.is_file():
        return ws.config_path
    if ws.root:
        legacy = ws.root / "configs" / "default.yaml"
        if legacy.is_file():
            return legacy
    return None


def default_settings() -> QSettings:
    """Application-scoped settings (user registry on Windows, plist on macOS, etc.)."""
    return QSettings("ModelBuilder", "ModelBuilderGUI")
