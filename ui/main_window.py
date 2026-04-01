"""
Main application window: sidebar navigation and stacked placeholder pages.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from mb import __version__ as MB_VERSION

from ui.pages import ConvertPage, DataPage, HomePage, InfoPage, TrainPage
from ui.workspace import Workspace, default_settings


class MainWindow(QMainWindow):
    NAV_ITEMS = [
        ("Home", HomePage),
        ("Data", DataPage),
        ("Train", TrainPage),
        ("Convert", ConvertPage),
        ("Info", InfoPage),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Model Builder")
        self.resize(960, 640)

        self._settings = default_settings()
        self._workspace = Workspace.load(self._settings)

        self._build_ui()
        self._apply_workspace_to_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._nav = QListWidget()
        self._nav.setFixedWidth(180)
        self._nav.setSpacing(6)
        for label, _page_cls in self.NAV_ITEMS:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 32))
            self._nav.addItem(item)
        self._nav.setCurrentRow(0)
        self._nav.currentRowChanged.connect(self._on_nav_changed)
        layout.addWidget(self._nav, 0)

        self._stack = QStackedWidget()
        for _label, page_cls in self.NAV_ITEMS:
            self._stack.addWidget(self._wrap_scroll(page_cls()))
        layout.addWidget(self._stack, 1)

        self._build_menu()

        self.statusBar().showMessage(self._status_text())

    def _wrap_scroll(self, page: QWidget) -> QScrollArea:
        """Wrap pages in a scroll area for smaller screens and dense forms."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        act_open = QAction("Set &workspace folder…", self)
        act_open.triggered.connect(self._choose_workspace)
        file_menu.addAction(act_open)
        act_cfg = QAction("Set &config file…", self)
        act_cfg.triggered.connect(self._choose_config)
        file_menu.addAction(act_cfg)
        file_menu.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        help_menu = self.menuBar().addMenu("&Help")
        act_about = QAction("&About Model Builder", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _on_nav_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)

    def _status_text(self) -> str:
        if self._workspace.root:
            root = str(self._workspace.root)
            cfg = (
                f" | Config: {self._workspace.config_path}"
                if self._workspace.config_path
                else ""
            )
            return f"Workspace: {root}{cfg}"
        return "No workspace folder set — use File → Set workspace folder"

    def _apply_workspace_to_ui(self) -> None:
        self.statusBar().showMessage(self._status_text())

    def _choose_workspace(self) -> None:
        start = str(self._workspace.root) if self._workspace.root else ""
        path = QFileDialog.getExistingDirectory(
            self,
            "Workspace root folder",
            start,
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        self._workspace.root = Path(path)
        self._workspace.save(self._settings)
        self._apply_workspace_to_ui()

    def _choose_config(self) -> None:
        start = str(self._workspace.config_path.parent) if self._workspace.config_path else ""
        if self._workspace.root and not start:
            start = str(self._workspace.root)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Optional YAML config",
            start,
            "YAML (*.yaml *.yml);;All files (*.*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        self._workspace.config_path = Path(path)
        self._workspace.save(self._settings)
        self._apply_workspace_to_ui()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Model Builder",
            f"<h3>Model Builder</h3>"
            f"<p>Desktop UI (PySide6) for the <code>mb</code> CLI and library.</p>"
            f"<p><b>Version:</b> {MB_VERSION}</p>"
            f"<p>CLI: <code>mb --help</code></p>",
        )

    def closeEvent(self, event) -> None:
        self._workspace.save(self._settings)
        super().closeEvent(event)
