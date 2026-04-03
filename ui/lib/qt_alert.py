"""
Qt message box helpers for Qt applications.
"""
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QWidget, QMessageBox

from mb.utils.translations import _


def _make_box(
    parent: Optional[QWidget],
    icon: QMessageBox.Icon,
    title: str,
    message: str,
    buttons: QMessageBox.StandardButton,
    default: QMessageBox.StandardButton,
) -> QMessageBox:
    """Create a QMessageBox with translated button labels and the given default."""
    box = QMessageBox(icon, title, message, buttons, parent)
    box.setDefaultButton(default)

    # Translate standard button labels
    _translations = {
        QMessageBox.StandardButton.Ok: _("OK"),
        QMessageBox.StandardButton.Cancel: _("Cancel"),
        QMessageBox.StandardButton.Yes: _("Yes"),
        QMessageBox.StandardButton.No: _("No"),
    }
    for btn_type, label in _translations.items():
        btn = box.button(btn_type)
        if btn is not None:
            btn.setText(label)

    return box


def qt_alert(
    parent: Optional[QWidget],
    title: str,
    message: str,
    kind: str = "info",
):
    """Show a Qt message box. kind: info, warning, error, askokcancel, askyesno, askyesnocancel."""
    if kind == "askokcancel":
        box = _make_box(
            parent, QMessageBox.Icon.Question, title, message,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        return box.exec() == QMessageBox.StandardButton.Ok
    if kind == "askyesno":
        box = _make_box(
            parent, QMessageBox.Icon.Question, title, message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return box.exec() == QMessageBox.StandardButton.Yes
    if kind == "askyesnocancel":
        box = _make_box(
            parent, QMessageBox.Icon.Question, title, message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        return box.exec()
    if kind == "error":
        box = _make_box(
            parent, QMessageBox.Icon.Critical, title, message,
            QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        box.exec()
        return None
    if kind == "warning":
        box = _make_box(
            parent, QMessageBox.Icon.Warning, title, message,
            QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        box.exec()
        return None
    # info
    box = _make_box(
        parent, QMessageBox.Icon.Information, title, message,
        QMessageBox.StandardButton.Ok,
        QMessageBox.StandardButton.Ok,
    )
    box.exec()
    return None


def qt_operation_error(
    parent: Optional[QWidget],
    title: str,
    summary: str,
    detail: Optional[str] = None,
    *,
    with_log_actions: bool = True,
) -> None:
    """
    Show a critical error with optional expandable details (e.g. full exception text).

    Prefer this over raw ``QMessageBox.critical`` for backend failures so users
    get a short summary plus technical detail when needed.

    When ``with_log_actions`` is True (default), adds **Copy details** (clipboard)
    and **Open log folder** (Model Builder log directory from
    :func:`mb.utils.logging_setup.get_log_directory`) in addition to **OK**.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(summary)
    if detail:
        box.setDetailedText(detail)

    copy_btn = None
    log_btn = None
    if with_log_actions:
        copy_btn = box.addButton(_("Copy details"), QMessageBox.ButtonRole.ActionRole)
        log_btn = box.addButton(_("Open log folder"), QMessageBox.ButtonRole.ActionRole)

    ok_btn = box.addButton(QMessageBox.StandardButton.Ok)
    box.setDefaultButton(ok_btn)
    ok_widget = box.button(QMessageBox.StandardButton.Ok)
    if ok_widget is not None:
        ok_widget.setText(_("OK"))

    box.exec()

    clicked = box.clickedButton()
    if copy_btn is not None and clicked == copy_btn:
        if detail:
            text = f"{summary}\n\n{detail}".strip()
        else:
            text = (summary or "").strip()
        if text:
            QGuiApplication.clipboard().setText(text)
    elif log_btn is not None and clicked == log_btn:
        try:
            from mb.utils.logging_setup import get_log_directory

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(get_log_directory())))
        except Exception:
            pass
