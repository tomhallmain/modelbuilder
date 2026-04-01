"""Information page scaffold for model/dataset queries."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class InfoPage(QWidget):
    """UI scaffold for `mb info model` and `mb info dataset`."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(QLabel("<h2>Info</h2>"))
        root.addWidget(QLabel("Inspect model metadata and dataset structure/statistics."))

        tabs = QTabWidget()
        tabs.addTab(self._build_model_tab(), "Model")
        tabs.addTab(self._build_dataset_tab(), "Dataset")
        root.addWidget(tabs, 1)

    def _build_model_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb info model")
        form = QFormLayout(group)
        self.model_path = QLineEdit()
        form.addRow("Model path", self._path_row(self.model_path))
        v.addWidget(group)

        actions = QHBoxLayout()
        self.btn_model_info = QPushButton("Inspect Model")
        self.btn_model_info.setEnabled(False)
        self.btn_model_info.setToolTip("Info callback not wired yet.")
        actions.addWidget(self.btn_model_info)
        actions.addStretch(1)
        v.addLayout(actions)

        self.model_output = QTextEdit()
        self.model_output.setReadOnly(True)
        self.model_output.setPlaceholderText("Model info output will appear here.")
        v.addWidget(self.model_output, 1)
        return tab

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb info dataset")
        form = QFormLayout(group)
        self.dataset_dir = QLineEdit("data")
        form.addRow("Data dir", self._path_row(self.dataset_dir))
        v.addWidget(group)

        actions = QHBoxLayout()
        self.btn_dataset_info = QPushButton("Inspect Dataset")
        self.btn_dataset_info.setEnabled(False)
        self.btn_dataset_info.setToolTip("Info callback not wired yet.")
        actions.addWidget(self.btn_dataset_info)
        actions.addStretch(1)
        v.addLayout(actions)

        self.dataset_output = QTextEdit()
        self.dataset_output.setReadOnly(True)
        self.dataset_output.setPlaceholderText("Dataset info output will appear here.")
        v.addWidget(self.dataset_output, 1)
        return tab

    def _path_row(self, edit: QLineEdit) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton("Browse...")
        browse.setEnabled(False)
        browse.setToolTip("File dialog callback not wired yet.")
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row
