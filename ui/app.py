"""
Application entry: QApplication and main window lifecycle.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.stderr.write(
            "PySide6 is required for the GUI. Install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "or: pip install -e .\n"
        )
        return 1

    from PySide6.QtCore import QThreadPool

    from mb.utils.translations import _
    from ui.main_window import MainWindow
    from utils.notification_manager import notification_manager

    app = QApplication(sys.argv)
    app.setApplicationName(_("Model Builder"))
    app.setOrganizationName("ModelBuilder")

    def _on_about_to_quit() -> None:
        """Cancel stray timers; best-effort wait for pool workers before process teardown."""
        notification_manager.cleanup_threads()
        QThreadPool.globalInstance().waitForDone(2_000)

    app.aboutToQuit.connect(_on_about_to_quit)

    win = MainWindow()
    win.show()
    return app.exec()
