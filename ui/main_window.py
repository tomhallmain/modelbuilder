"""
Main application window: sidebar navigation and stacked placeholder pages.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, QThreadPool, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
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

from ui.app_actions import AppActions
from ui.app_theme import apply_theme
from ui.controllers.cache_controller import CacheController
from ui.main_thread_bridge import MainThreadBridge
from ui.controllers.notification_controller import NotificationController
from ui.lib.qt_alert import qt_alert
from ui.pages import ConfigPage, ConvertPage, DataPage, HomePage, InfoPage, TrainPage
from ui.workspace import Workspace, default_settings, effective_pipeline_config_path
from mb.pipeline_config import reload_pipeline_config

from utils.config import (
    DEFAULT_APPLICATION_YAML,
    get_application_config,
    reload_application_config,
)
from utils.logging_setup import apply_application_log_settings
from utils.translations import apply_application_locale
from utils.notification_manager import notification_manager


class MainWindow(QMainWindow):
    NAV_ITEMS = [
        ("Home", HomePage),
        ("Data", DataPage),
        ("Train", TrainPage),
        ("Convert", ConvertPage),
        ("Config", ConfigPage),
        ("Info", InfoPage),
    ]

    #: Used by :class:`NotificationController` for title restore (stable caption).
    window_id: int = 0

    def __init__(self) -> None:
        super().__init__()
        self._base_window_title = "Model Builder"
        self.setWindowTitle(self._base_window_title)

        self._settings = default_settings()
        self._workspace = Workspace.load(self._settings)
        reload_application_config(self._effective_application_config_path())
        reload_pipeline_config(self._effective_pipeline_config_path())
        apply_theme(QApplication.instance())
        apply_application_log_settings()
        apply_application_locale(verbose=False)
        self._apply_main_window_geometry_from_config()

        self._thread_bridge = MainThreadBridge(self)
        self._notifications = NotificationController(self)
        self.app_actions = self._build_app_actions()
        notification_manager.set_app_actions(self.app_actions, window_id=self.window_id)

        self._page_widgets: list[QWidget] = []
        self._build_ui()
        self._cache = CacheController(self)
        self._cache.load_info_cache()
        self._run_page_startup_validation()
        self._cache.start_periodic_store()
        self._apply_workspace_to_ui()

    @property
    def nav_widget(self) -> QListWidget:
        return self._nav

    @property
    def page_widgets(self) -> list[QWidget]:
        return self._page_widgets

    def reload_mb_yaml_config(self) -> None:
        """Reload application + pipeline YAML (e.g. after user picks a config file)."""
        reload_application_config(self._effective_application_config_path(), force=True)
        reload_pipeline_config(self._effective_pipeline_config_path(), force=True)
        self.refresh_application_shell_settings()

    def refresh_application_shell_settings(self) -> None:
        """Re-apply theme, logging, locale, window geometry, and cache interval from config."""
        apply_theme(QApplication.instance())
        apply_application_log_settings()
        apply_application_locale(verbose=False)
        self._apply_main_window_geometry_from_config()
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.restart_periodic_store()

    def _effective_application_config_path(self) -> Path | None:
        """
        ``gui`` / ``app`` YAML: explicit file from **File → Set config file…**, else
        ``configs/application.yaml`` or legacy ``configs/default.yaml`` under workspace
        (packaged defaults come from ``mb/config/application.example.yaml`` when no file is set).
        """
        ws = self._workspace
        if ws.config_path is not None:
            return ws.config_path
        if ws.root:
            for name in ("application.yaml", "default.yaml"):
                candidate = ws.root / "configs" / name
                if candidate.is_file():
                    return candidate
        return None

    def _effective_pipeline_config_path(self) -> Path | None:
        """
        Pipeline YAML: ``configs/pipeline.yaml`` under workspace, else the same file
        as the application path when it is a legacy combined ``default.yaml``, else
        packaged defaults.
        """
        return effective_pipeline_config_path(self._workspace)

    def _apply_main_window_geometry_from_config(self) -> None:
        size = get_application_config().gui.default_main_window_size
        w, h = 960, 640
        if isinstance(size, str) and "x" in size.lower():
            try:
                parts = size.lower().split("x", 1)
                w, h = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                pass
        self.resize(max(400, w), max(300, h))

    def get_title_from_base_dir(self) -> str:
        """Stable window title for toast/title_notify restore (not workspace path)."""
        return self._base_window_title

    def _build_app_actions(self) -> AppActions:
        """Wire actions; GUI-touching callables use :class:`MainThreadBridge` where needed.

        ``NotificationController.toast`` / ``title_notify`` already marshal via Qt
        signals. ``title`` and ``alert`` are wrapped so ``notification_manager``
        timers and any worker-thread callers are safe.
        """
        nc = self._notifications
        ts = self._thread_bridge.wrap

        return AppActions(
            {
                "get_window": lambda: self,
                "toast": nc.toast,
                "_alert": ts(nc.alert),
                "title_notify": nc.title_notify,
                "refresh": lambda: None,
                "title": ts(self.setWindowTitle),
            },
            master=self,
        )

    def _run_page_startup_validation(self) -> None:
        """
        Run each page's validation once after constructor + cache restore.

        Without this, pages validate in ``__init__`` and again in
        ``restore_gui_state`` (and signal handlers like ``currentChanged``),
        doubling ``[invalid]`` / ``[info]`` lines on startup.
        """
        for w in self._page_widgets:
            fn = getattr(w, "_run_startup_validation", None)
            if callable(fn):
                fn()

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
        self._stack.setObjectName("main_nav_stack")
        for _label, page_cls in self.NAV_ITEMS:
            page = page_cls()
            self._page_widgets.append(page)
            self._stack.addWidget(self._wrap_scroll(page))
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
        act_app_yaml = QAction("Open &application settings (YAML)…", self)
        act_app_yaml.triggered.connect(self._open_application_settings_yaml)
        act_app_yaml.setToolTip(
            "Desktop shell options (toasts, cache interval, …); see mb/config/application.example.yaml. "
            "Reload the app or use Set config file for pipeline YAML."
        )
        file_menu.addAction(act_app_yaml)
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

    def _open_application_settings_yaml(self) -> None:
        """Open packaged ``mb/config/application.example.yaml`` in the OS default editor / viewer."""
        path = DEFAULT_APPLICATION_YAML
        if not path.is_file():
            qt_alert(
                self,
                "Application settings",
                f"File not found:\n{path}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

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
        self.reload_mb_yaml_config()
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
        self.reload_mb_yaml_config()
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
        pool = QThreadPool.globalInstance()
        if pool.activeThreadCount() > 0:
            answer = QMessageBox.warning(
                self,
                "Background task",
                "A background task is still running. Closing now may leave partial outputs "
                "(checkpoints, datasets, copies). Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.No:
                event.ignore()
                return
        self._cache.stop_periodic_store()
        self._cache.store_info_cache(sync=True)
        self._workspace.save(self._settings)
        super().closeEvent(event)
