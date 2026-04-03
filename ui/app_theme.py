"""
Global application styling: dark base with orange/amber accents (construction-style).

Apply on QApplication so windows inherit the look. Palette and stylesheet are built
from :data:`COLORS` and optional overrides in application config (see
:func:`resolve_theme_colors`).
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

_DEFAULT_TOAST_WARNING = "#5D4037"
_DEFAULT_TOAST_SUCCESS = "#33691E"


@dataclass(frozen=True)
class ThemeColors:
    """Palette for documentation and :func:`apply_theme`."""

    bg_deep: str = "#121212"
    bg_window: str = "#1a1a1a"
    bg_elevated: str = "#242424"
    bg_input: str = "#2c2c2c"
    border: str = "#3f3f3f"
    border_focus: str = "#E65100"
    text: str = "#ececec"
    text_muted: str = "#9e9e9e"
    accent_orange: str = "#F57C00"
    accent_deep: str = "#E65100"
    accent_amber: str = "#FFC107"
    highlight: str = "#FF8F00"
    highlight_text: str = "#1a1a1a"
    selection_alt: str = "#4E342E"


COLORS = ThemeColors()


def _qcolor(hex_rgb: str) -> QColor:
    return QColor(hex_rgb)


def _hex(c: QColor) -> str:
    return c.name(QColor.NameFormat.HexRgb)


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() * (1 - t) + b.red() * t),
        int(a.green() * (1 - t) + b.green() * t),
        int(a.blue() * (1 - t) + b.blue() * t),
    )


def resolve_theme_colors() -> ThemeColors:
    """
    Effective theme: :data:`COLORS` merged with ``gui`` color keys from application config.

    Null / missing keys fall back to :data:`COLORS`. Accents drive focus rings, highlights,
    and link/hover tones via ``accent_color`` / ``accent_secondary_color``.
    """
    try:
        from utils.config import get_application_config

        gui = get_application_config().gui
    except Exception:
        return COLORS

    base = COLORS
    bg_s = gui.background_color
    fg_s = gui.foreground_color

    qc_bg = _qcolor(str(bg_s)) if bg_s else _qcolor(base.bg_window)
    qc_fg = _qcolor(str(fg_s)) if fg_s else _qcolor(base.text)
    qc_muted = _blend(qc_fg, qc_bg, 0.52)

    ap = gui.accent_color
    asec = gui.accent_secondary_color
    accent_orange_hex = str(ap) if ap else base.accent_orange
    accent_amber_hex = str(asec) if asec else base.accent_amber
    qc_accent_o = _qcolor(accent_orange_hex)
    qc_accent_a = _qcolor(accent_amber_hex)
    accent_deep_hex = _hex(qc_accent_o.darker(112))
    highlight_hex = _hex(_blend(qc_accent_o, qc_accent_a, 0.35))

    return ThemeColors(
        bg_window=_hex(qc_bg),
        bg_deep=_hex(qc_bg.darker(118)),
        bg_elevated=_hex(qc_bg.lighter(108)),
        bg_input=_hex(qc_bg.lighter(112)),
        border=_hex(_blend(qc_bg, qc_fg, 0.22)),
        border_focus=accent_deep_hex,
        text=_hex(qc_fg),
        text_muted=_hex(qc_muted),
        accent_orange=accent_orange_hex,
        accent_deep=accent_deep_hex,
        accent_amber=accent_amber_hex,
        highlight=highlight_hex,
        highlight_text=base.highlight_text,
        selection_alt=base.selection_alt,
    )


def theme_font_point_size() -> int:
    try:
        from utils.config import get_application_config

        n = int(get_application_config().gui.font_size)
        return max(6, min(24, n))
    except Exception:
        return 8


def style_foreground_hex() -> str:
    return resolve_theme_colors().text


def style_elevated_bg_hex() -> str:
    return resolve_theme_colors().bg_elevated


def toast_warning_background() -> str:
    try:
        from utils.config import get_application_config

        v = get_application_config().gui.toast_color_warning
        if v:
            return str(v)
    except Exception:
        pass
    return _DEFAULT_TOAST_WARNING


def toast_success_background() -> str:
    try:
        from utils.config import get_application_config

        v = get_application_config().gui.toast_color_success
        if v:
            return str(v)
    except Exception:
        pass
    return _DEFAULT_TOAST_SUCCESS


class AppStyle:
    """
    Live tokens from application config + theme resolution.

    Prefer these helpers over hard-coded hex so toasts and custom widgets stay in sync.
    """

    @staticmethod
    def toast_warning_bg() -> str:
        return toast_warning_background()

    @staticmethod
    def toast_success_bg() -> str:
        return toast_success_background()


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


def _stylesheet(c: ThemeColors, font_pt: int) -> str:
    return f"""
    * {{
        outline: none;
    }}
    QWidget {{
        background-color: {c.bg_window};
        color: {c.text};
        selection-background-color: {c.highlight};
        selection-color: {c.highlight_text};
        font-size: {font_pt}pt;
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
    Apply Fusion style, palette, and global stylesheet.

    When ``colors`` is omitted, uses :func:`resolve_theme_colors` and
    :func:`theme_font_point_size` from application config.
    """
    c = colors or resolve_theme_colors()
    font_pt = theme_font_point_size()
    app.setStyle("Fusion")
    _apply_fusion_palette(app, c)
    app.setStyleSheet(_stylesheet(c, font_pt))
