"""Helpers to re-apply gettext strings to :class:`PySide6.QtWidgets.QFormLayout` label cells."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout


def apply_qform_label_column(form: QFormLayout, labels: list[str]) -> None:
    """Set the label widget text for rows 0..len(labels)-1 (must match layout construction order)."""
    for row, text in enumerate(labels):
        if row >= form.rowCount():
            break
        item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        if item is None:
            continue
        w = item.widget()
        if w is not None:
            w.setText(text)
