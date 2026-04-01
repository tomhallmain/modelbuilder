"""Training page scaffold aligned with `mb train` arguments."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainPage(QWidget):
    """UI scaffold for training configuration."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(QLabel("<h2>Train</h2>"))
        root.addWidget(QLabel("Configure framework, architecture, paths, and hyperparameters for `mb train`."))

        core_group = QGroupBox("Core configuration")
        core_form = QFormLayout(core_group)
        self.model_type = QComboBox()
        self.model_type.addItems(["image_classification"])
        self.framework = QComboBox()
        self.framework.addItems(["pytorch", "keras"])
        self.architecture = QLineEdit("resnet34")
        self.data_dir = QLineEdit("data")
        self.output_dir = QLineEdit("data/models")
        self.resume_from = QLineEdit()
        self.run_id = QLineEdit()
        self.skip_snapshot = QCheckBox("Skip unified snapshot update")

        core_form.addRow("Model type", self.model_type)
        core_form.addRow("Framework", self.framework)
        core_form.addRow("Architecture", self.architecture)
        core_form.addRow("Data dir", self._path_row(self.data_dir))
        core_form.addRow("Output dir", self._path_row(self.output_dir))
        core_form.addRow("Resume checkpoint", self._path_row(self.resume_from))
        core_form.addRow("Run ID (optional)", self.run_id)
        core_form.addRow("", self.skip_snapshot)
        root.addWidget(core_group)

        hp_group = QGroupBox("Hyperparameters")
        hp_form = QFormLayout(hp_group)
        self.frozen_epochs = QSpinBox()
        self.frozen_epochs.setRange(0, 10000)
        self.frozen_epochs.setValue(5)
        self.unfrozen_epochs = QSpinBox()
        self.unfrozen_epochs.setRange(0, 10000)
        self.unfrozen_epochs.setValue(20)
        self.frozen_lr = self._lr_spin(0.001)
        self.unfrozen_lr_max = self._lr_spin(0.0003)
        self.unfrozen_lr_min = self._lr_spin(0.00001)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(0, 8192)
        self.batch_size.setSpecialValueText("Auto")
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 4096)
        self.image_size.setValue(224)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 128)
        self.num_workers.setSpecialValueText("Config default")

        hp_form.addRow("Frozen epochs", self.frozen_epochs)
        hp_form.addRow("Unfrozen epochs", self.unfrozen_epochs)
        hp_form.addRow("Frozen LR", self.frozen_lr)
        hp_form.addRow("Unfrozen LR max", self.unfrozen_lr_max)
        hp_form.addRow("Unfrozen LR min", self.unfrozen_lr_min)
        hp_form.addRow("Batch size", self.batch_size)
        hp_form.addRow("Image size", self.image_size)
        hp_form.addRow("Num workers", self.num_workers)
        root.addWidget(hp_group)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton("Validate Training Config")
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setEnabled(False)
        self.btn_start.setToolTip("Training callback not wired yet.")
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_start)
        actions.addStretch(1)
        root.addLayout(actions)
        root.addStretch(1)

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

    def _lr_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(8)
        spin.setRange(0.0, 10.0)
        spin.setSingleStep(0.0001)
        spin.setValue(value)
        return spin
