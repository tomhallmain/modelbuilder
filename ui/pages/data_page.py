"""Data operations page mirroring `mb data` subcommands."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class DataPage(QWidget):
    """UI scaffold for data command forms."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("<h2>Data</h2>")
        root.addWidget(title)
        root.addWidget(QLabel("Prepare datasets using gather, convert, deduplicate, upscale, and split flows."))

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_gather_tab(), "Gather")
        self.tabs.addTab(self._build_convert_tab(), "Convert")
        self.tabs.addTab(self._build_dedup_tab(), "Deduplicate")
        self.tabs.addTab(self._build_upscale_tab(), "Upscale")
        self.tabs.addTab(self._build_dataset_tab(), "Create Dataset")
        root.addWidget(self.tabs, 1)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton("Validate Inputs")
        self.btn_run = QPushButton("Run Data Command")
        self.btn_run.setEnabled(False)
        self.btn_run.setToolTip("Execution callbacks pending.")
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_run)
        actions.addStretch(1)
        root.addLayout(actions)

    def _build_gather_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data gather")
        form = QFormLayout(group)
        self.gather_source = QLineEdit()
        self.gather_subdirs = QLineEdit()
        self.gather_target_count = QSpinBox()
        self.gather_target_count.setRange(1, 5_000_000)
        self.gather_target_count.setValue(16000)
        self.gather_target_dir = QLineEdit("raw_data/coherent")
        self.gather_rejected_dir = QLineEdit()
        self.gather_subdir_weights = QLineEdit()
        self.gather_raw_data_dir = QLineEdit("raw_data")

        form.addRow("Source dir", self._path_row(self.gather_source))
        form.addRow("Subdirs (space-separated)", self.gather_subdirs)
        form.addRow("Target count", self.gather_target_count)
        form.addRow("Target dir", self._path_row(self.gather_target_dir))
        form.addRow("Rejected dir", self._path_row(self.gather_rejected_dir))
        form.addRow("Subdir weights", self.gather_subdir_weights)
        form.addRow("Raw data dir", self._path_row(self.gather_raw_data_dir))
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_convert_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data convert")
        form = QFormLayout(group)
        self.convert_raw_data_dir = QLineEdit("raw_data")
        self.convert_format = QLineEdit("jpeg")
        form.addRow("Raw data dir", self._path_row(self.convert_raw_data_dir))
        form.addRow("Format (jpeg/jpg)", self.convert_format)
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_dedup_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data deduplicate")
        form = QFormLayout(group)
        self.dedup_raw_data_dir = QLineEdit("raw_data")
        form.addRow("Raw data dir", self._path_row(self.dedup_raw_data_dir))
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_upscale_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data upscale")
        form = QFormLayout(group)
        self.upscale_raw_data_dir = QLineEdit("raw_data")
        self.upscale_review_dir = QLineEdit()
        form.addRow("Raw data dir", self._path_row(self.upscale_raw_data_dir))
        form.addRow("Review dir (optional)", self._path_row(self.upscale_review_dir))
        v.addWidget(group)
        v.addStretch(1)
        return tab

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb data create-dataset")
        form = QFormLayout(group)
        self.dataset_raw_data_dir = QLineEdit("raw_data")
        self.dataset_data_dir = QLineEdit("data")
        self.dataset_test_per_class = QSpinBox()
        self.dataset_test_per_class.setRange(1, 1_000_000)
        self.dataset_test_per_class.setValue(1000)
        self.dataset_seed = QSpinBox()
        self.dataset_seed.setRange(0, 2_147_483_647)
        self.dataset_run_id = QLineEdit()
        self.dataset_max_train = QSpinBox()
        self.dataset_max_train.setRange(0, 1_000_000)
        self.dataset_max_train.setSpecialValueText("None")
        self.dataset_balance_train = QCheckBox("Balance train set to smallest class")
        self.dataset_allow_external = QCheckBox("Allow external/removable storage")

        form.addRow("Raw data dir", self._path_row(self.dataset_raw_data_dir))
        form.addRow("Output data dir", self._path_row(self.dataset_data_dir))
        form.addRow("Test images per class", self.dataset_test_per_class)
        form.addRow("Seed (optional)", self.dataset_seed)
        form.addRow("Run ID (optional)", self.dataset_run_id)
        form.addRow("Max train per class", self.dataset_max_train)
        form.addRow("", self.dataset_balance_train)
        form.addRow("", self.dataset_allow_external)
        v.addWidget(group)
        v.addStretch(1)
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
