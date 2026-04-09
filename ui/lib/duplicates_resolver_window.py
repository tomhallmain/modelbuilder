"""Simple duplicates resolver dialog for list-only deduplication runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from mb.utils.translations import _


class DuplicatesResolverDialog(QDialog):
    """Show duplicate groups and allow copy/remove actions."""

    def __init__(self, duplicate_items: List[dict], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("duplicates_resolver_dialog")
        self.setWindowTitle(_("Duplicate Resolver"))
        self.resize(900, 620)
        self._groups = self._group_items(duplicate_items)

        root = QVBoxLayout(self)
        intro = QLabel(
            _(
                "Review potential duplicates from snapshot metadata. Select one path and copy it, "
                "or remove the non-selected peer when exactly two files remain in a group."
            )
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        for group_id in sorted(self._groups.keys()):
            container_layout.addWidget(self._build_group_box(group_id))
        container_layout.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

    def _group_items(self, duplicate_items: List[dict]) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = {}
        for item in duplicate_items:
            ids = item.get("duplicate_group_ids") or []
            if not ids:
                continue
            for group_id in ids:
                grouped.setdefault(str(group_id), []).append(item)
        return grouped

    def _build_group_box(self, group_id: str) -> QGroupBox:
        items = self._groups.get(group_id, [])
        box = QGroupBox(_("{gid} ({count} files)").format(gid=group_id, count=len(items)))
        layout = QVBoxLayout(box)

        list_widget = QListWidget()
        for item in items:
            abs_path = str(item.get("absolute_converted_path") or "")
            text = abs_path or str(item.get("converted_path") or "")
            row = QListWidgetItem(text)
            row.setData(Qt.ItemDataRole.UserRole, item)
            list_widget.addItem(row)
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)

        button_row = QHBoxLayout()
        copy_btn = QPushButton(_("Copy selected path"))
        copy_btn.clicked.connect(lambda: self._copy_selected_path(list_widget))
        remove_btn = QPushButton(_("Remove non-selected peer"))
        remove_btn.clicked.connect(lambda: self._remove_non_selected_peer(group_id, list_widget, box))
        button_row.addWidget(copy_btn)
        button_row.addWidget(remove_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return box

    def _selected_item_data(self, list_widget: QListWidget) -> dict | None:
        row = list_widget.currentItem()
        if row is None:
            return None
        data = row.data(Qt.ItemDataRole.UserRole)
        return data if isinstance(data, dict) else None

    def _copy_selected_path(self, list_widget: QListWidget) -> None:
        data = self._selected_item_data(list_widget)
        if not data:
            QMessageBox.information(self, _("No selection"), _("Select a file path first."))
            return
        text = str(data.get("absolute_converted_path") or data.get("converted_path") or "")
        if not text:
            QMessageBox.information(self, _("Unavailable"), _("Selected item has no path."))
            return
        QApplication.clipboard().setText(text)

    def _remove_non_selected_peer(self, group_id: str, list_widget: QListWidget, box: QGroupBox) -> None:
        if list_widget.count() != 2:
            QMessageBox.information(
                self,
                _("Action not allowed"),
                _("This action requires exactly two files in the group."),
            )
            return
        selected = self._selected_item_data(list_widget)
        if not selected:
            QMessageBox.information(self, _("No selection"), _("Select the file you want to keep."))
            return
        selected_path = str(selected.get("absolute_converted_path") or "")
        peer_row = None
        for idx in range(list_widget.count()):
            item = list_widget.item(idx)
            data = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(data, dict):
                continue
            candidate = str(data.get("absolute_converted_path") or "")
            if candidate != selected_path:
                peer_row = idx
                break
        if peer_row is None:
            QMessageBox.information(self, _("Unavailable"), _("Could not resolve peer file path."))
            return
        peer_item = list_widget.item(peer_row)
        peer_data = peer_item.data(Qt.ItemDataRole.UserRole) or {}
        peer_path = Path(str(peer_data.get("absolute_converted_path") or ""))
        if not peer_path:
            QMessageBox.information(self, _("Unavailable"), _("Peer file path is empty."))
            return
        if not peer_path.exists():
            QMessageBox.warning(self, _("Missing file"), _("Peer file no longer exists:\n{p}").format(p=peer_path))
            list_widget.takeItem(peer_row)
            box.setTitle(_("{gid} ({count} files)").format(gid=group_id, count=list_widget.count()))
            return
        peer_path.unlink()
        list_widget.takeItem(peer_row)
        box.setTitle(_("{gid} ({count} files)").format(gid=group_id, count=list_widget.count()))
