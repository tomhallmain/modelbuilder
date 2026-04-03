"""
Global application styling: dark base with orange/amber accents (construction-style).

Apply once on QApplication so all windows, dialogs, menus, and native-style
widgets (QMessageBox, QFileDialog, etc.) inherit the same look.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeColors:
    """Palette for documentation and optional use in custom widgets."""

    bg_deep: str = "#121212"
    bg_window: str = "#1a1a1a"
    bg_elevated: str = "#242424"
    bg_input: str = "#2c2c2c"
    border: str = "#3f3f3f"
    border_focus: str = "#E65100"
    text: str = "#ececec"
    text_muted: str = "#9e9e9e"
    # Home-depot-ish orange + construction amber
    accent_orange: str = "#F57C00"
    accent_deep: str = "#E65100"
    accent_amber: str = "#FFC107"
    highlight: str = "#FF8F00"
    highlight_text: str = "#1a1a1a"
    selection_alt: str = "#4E342E"


COLORS = ThemeColors()


class AppStyle:
    """
    Stable color tokens for widgets that need explicit hex (toasts, overlays).

    Aligns with :data:`COLORS` / :func:`apply_theme` so notification UI matches the app theme.
    """

    BG_COLOR = COLORS.bg_elevated
    FG_COLOR = COLORS.text
    TOAST_COLOR_WARNING = "#5D4037"
    TOAST_COLOR_SUCCESS = "#33691E"


def _qcolor(hex_rgb: str) -> QColor:
    return QColor(hex_rgb)


def _apply_fusion_palette(app: QApplication, c: ThemeColors) -> None:
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, _qcolor(c.bg_window))
    pal.setColor(QPalette.ColorRole.WindowText, _qcolor(c.text))
    pal.setColor(QPalette.ColorRole.Base, _qcolor(c.bg_input))
    pal.setColor(QPalette.ColorRole.AlternateBase, _qcolor(c.bg_elevated))
    pal.setColor(QPalette.ColorRole.ToolTipBase, _qcolor(c.bg_elevated))
    pal.setColor(QPalette.ColorRole.ToolTipText, _qcolor(c.text))
    pal.setColor(QPalette.ColorRole.Text, _qcolor(c.text))
    pal.setColor(QPalette.ColorRole.Button, _qcolor(c.bg_elevated))
    pal.setColor(QPalette.ColorRole.ButtonText, _qcolor(c.text))
    pal.setColor(QPalette.ColorRole.BrightText, _qcolor(c.accent_amber))
    pal.setColor(QPalette.ColorRole.Link, _qcolor(c.accent_amber))
    pal.setColor(QPalette.ColorRole.Highlight, _qcolor(c.highlight))
    pal.setColor(QPalette.ColorRole.HighlightedText, _qcolor(c.highlight_text))
    pal.setColor(QPalette.ColorRole.PlaceholderText, _qcolor(c.text_muted))
    app.setPalette(pal)


def _stylesheet(c: ThemeColors) -> str:
    # Global QWidget color helps dialogs and message boxes inherit text/background.
    return f"""
    * {{
        outline: none;
    }}
    QWidget {{
        background-color: {c.bg_window};
        color: {c.text};
        selection-background-color: {c.highlight};
        selection-color: {c.highlight_text};
    }}
    QMainWindow, QDialog, QMessageBox {{
        background-color: {c.bg_window};
    }}
    QMenuBar {{
        background-color: {c.bg_deep};
        color: {c.text};
        border-bottom: 1px solid {c.border};
        padding: 2px;
    }}
    QMenuBar::item:selected {{
        background-color: {c.bg_elevated};
        border-radius: 3px;
    }}
    QMenuBar::item:pressed {{
        background-color: {c.selection_alt};
    }}
    QMenu {{
        background-color: {c.bg_elevated};
        color: {c.text};
        border: 1px solid {c.border};
        padding: 4px;
    }}
    QMenu::item:selected {{
        background-color: {c.selection_alt};
        border-left: 3px solid {c.accent_amber};
    }}
    QMenu::separator {{
        height: 1px;
        background: {c.border};
        margin: 4px 8px;
    }}
    QStatusBar {{
        background-color: {c.bg_deep};
        color: {c.text_muted};
        border-top: 1px solid {c.border};
    }}
    QStatusBar::item {{
        border: none;
    }}
    QLabel {{
        color: {c.text};
        background: transparent;
    }}
    QLabel a {{
        color: {c.accent_amber};
    }}
    QFrame {{
        border: none;
    }}
    QListWidget {{
        background-color: {c.bg_elevated};
        color: {c.text};
        border: 1px solid {c.border};
        border-radius: 4px;
        padding: 4px;
    }}
    QListWidget::item {{
        padding: 8px 10px;
        border-radius: 3px;
    }}
    QListWidget::item:hover {{
        background-color: {c.selection_alt};
    }}
    QListWidget::item:selected {{
        background-color: {c.selection_alt};
        color: {c.accent_amber};
        border-left: 3px solid {c.accent_orange};
    }}
    QStackedWidget {{
        background-color: {c.bg_window};
    }}
    QScrollBar:vertical {{
        background: {c.bg_deep};
        width: 12px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c.bg_elevated};
        min-height: 24px;
        border-radius: 4px;
        border: 1px solid {c.border};
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.selection_alt};
    }}
    QScrollBar:horizontal {{
        background: {c.bg_deep};
        height: 12px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c.bg_elevated};
        min-width: 24px;
        border-radius: 4px;
    }}
    QPushButton {{
        background-color: {c.bg_elevated};
        color: {c.text};
        border: 1px solid {c.border};
        border-radius: 4px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{
        border-color: {c.accent_orange};
        color: {c.accent_amber};
    }}
    QPushButton:pressed {{
        background-color: {c.selection_alt};
    }}
    QPushButton:default {{
        border: 2px solid {c.accent_orange};
    }}
    QLineEdit, QPlainTextEdit, QTextEdit {{
        background-color: {c.bg_input};
        color: {c.text};
        border: 1px solid {c.border};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border-color: {c.border_focus};
    }}
    QComboBox {{
        background-color: {c.bg_input};
        color: {c.text};
        border: 1px solid {c.border};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QComboBox:hover {{
        border-color: {c.accent_orange};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.bg_elevated};
        color: {c.text};
        selection-background-color: {c.selection_alt};
        selection-color: {c.accent_amber};
    }}
    QDialogButtonBox QPushButton {{
        min-width: 72px;
    }}
    QToolTip {{
        background-color: {c.bg_elevated};
        color: {c.text};
        border: 1px solid {c.accent_deep};
    }}
    QProgressBar {{
        border: 1px solid {c.border};
        border-radius: 4px;
        text-align: center;
        background-color: {c.bg_input};
        color: {c.text};
    }}
    QProgressBar::chunk {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c.accent_deep}, stop:1 {c.accent_amber});
        border-radius: 3px;
    }}
    """


def apply_theme(app: QApplication, colors: ThemeColors | None = None) -> None:
    """
    Apply Fusion style, dark palette, and global stylesheet to the application.

    Call once after creating QApplication and before showing any window.
    """
    c = colors or COLORS
    app.setStyle("Fusion")
    _apply_fusion_palette(app, c)
    app.setStyleSheet(_stylesheet(c))
