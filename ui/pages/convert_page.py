"""Model conversion page scaffold aligned with `mb convert`."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ConvertPage(QWidget):
    """UI scaffold for model conversion settings."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(QLabel("<h2>Convert</h2>"))
        root.addWidget(QLabel("Prepare conversion jobs for ONNX and SafeTensors targets."))

        group = QGroupBox("mb convert")
        form = QFormLayout(group)
        self.input_model = QLineEdit()
        self.output_model = QLineEdit()
        self.framework = QComboBox()
        self.framework.addItems(["auto-detect", "pytorch", "keras"])
        self.target = QComboBox()
        self.target.addItems(["onnx", "safetensors"])
        self.architecture = QLineEdit()
        self.num_classes = QSpinBox()
        self.num_classes.setRange(0, 1_000_000)
        self.num_classes.setSpecialValueText("Required only for PyTorch -> ONNX")
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 4096)
        self.image_size.setValue(224)

        form.addRow("Input model", self._path_row(self.input_model))
        form.addRow("Output model", self._path_row(self.output_model))
        form.addRow("Source framework", self.framework)
        form.addRow("Target format", self.target)
        form.addRow("Architecture", self.architecture)
        form.addRow("Num classes", self.num_classes)
        form.addRow("Image size", self.image_size)
        root.addWidget(group)

        hint = QLabel("Note: architecture/num_classes are needed for PyTorch -> ONNX.")
        hint.setWordWrap(True)
        root.addWidget(hint)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton("Validate Conversion")
        self.btn_convert = QPushButton("Run Conversion")
        self.btn_convert.setEnabled(False)
        self.btn_convert.setToolTip("Conversion callback not wired yet.")
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_convert)
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
