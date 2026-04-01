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
            "PySide6 is required for the GUI. Install with:\n"
            "  pip install modelbuilder[gui]\n"
            "or: pip install -r requirements-gui.txt\n"
        )
        return 1

    from ui.app_theme import apply_theme
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Model Builder")
    app.setOrganizationName("ModelBuilder")
    apply_theme(app)

    win = MainWindow()
    win.show()
    return app.exec()
