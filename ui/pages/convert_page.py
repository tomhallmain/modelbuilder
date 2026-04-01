"""Model conversion page scaffold aligned with `mb convert`."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.conversion.converters import convert_model, detect_model_framework
from ui.task_runner import start_task


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

        form.addRow("Input model", self._path_row(self.input_model, select_dir=False, save=False))
        form.addRow("Output model", self._path_row(self.output_model, select_dir=False, save=True))
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
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_convert)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Conversion validation and execution messages will appear here.")
        root.addWidget(self.output, 1)

        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_convert.clicked.connect(self._run_conversion)
        self._validate_inputs()

    def _path_row(self, edit: QLineEdit, select_dir: bool = True, save: bool = False) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse(edit, select_dir=select_dir, save=save))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, select_dir: bool = True, save: bool = False) -> None:
        start = edit.text().strip() or str(Path.cwd())
        if select_dir:
            value = QFileDialog.getExistingDirectory(
                self,
                "Select directory",
                start,
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)
        elif save:
            value, _ = QFileDialog.getSaveFileName(
                self,
                "Select output file",
                start,
                "Model files (*.onnx *.safetensors);;All files (*.*)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)
        else:
            value, _ = QFileDialog.getOpenFileName(
                self,
                "Select model file",
                start,
                "Model files (*.pth *.pt *.h5 *.keras *.onnx *.safetensors);;All files (*.*)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)
        self._validate_inputs()

    def _append(self, msg: str) -> None:
        self.output.append(msg)

    def _set_busy(self, busy: bool) -> None:
        self.btn_validate.setEnabled(not busy)
        self.btn_convert.setEnabled(not busy and self._can_run())

    def _collect_inputs(self) -> dict:
        input_path = Path(self.input_model.text().strip())
        output_path = Path(self.output_model.text().strip())
        if not input_path.exists():
            raise ValueError("Input model file does not exist.")
        if not output_path.parent.exists():
            raise ValueError("Output model parent directory does not exist.")

        framework = self.framework.currentText()
        source_framework = None if framework == "auto-detect" else framework
        target = self.target.currentText()
        architecture = self.architecture.text().strip() or None
        num_classes = int(self.num_classes.value()) if self.num_classes.value() > 0 else None

        detected = source_framework or detect_model_framework(input_path)
        if detected == "pytorch" and target == "onnx":
            if architecture is None or num_classes is None:
                raise ValueError("PyTorch -> ONNX requires architecture and num classes.")

        return {
            "input_path": input_path,
            "output_path": output_path,
            "source_framework": source_framework,
            "target_format": target,
            "architecture": architecture,
            "num_classes": num_classes,
            "image_size": int(self.image_size.value()),
        }

    def _can_run(self) -> bool:
        try:
            self._collect_inputs()
            return True
        except ValueError:
            return False

    def _validate_inputs(self) -> None:
        try:
            payload = self._collect_inputs()
            detected = payload["source_framework"] or detect_model_framework(payload["input_path"])
            self.btn_convert.setEnabled(True)
            self._append(f"[ok] conversion inputs valid (source={detected or 'unknown'})")
        except ValueError as exc:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setToolTip(str(exc))
            self._append(f"[invalid] {exc}")

    def _run_conversion(self) -> None:
        payload = self._collect_inputs()
        self._append(f"[run] mb convert {payload['input_path'].name} -> {payload['target_format']}")
        self._set_busy(True)
        start_task(
            self._execute_conversion,
            self._on_success,
            self._on_error,
            lambda: self._set_busy(False),
            payload,
        )

    def _execute_conversion(self, payload: dict) -> bool:
        return bool(
            convert_model(
                input_path=payload["input_path"],
                output_path=payload["output_path"],
                source_framework=payload["source_framework"],
                target_format=payload["target_format"],
                architecture=payload["architecture"],
                num_classes=payload["num_classes"],
                image_size=payload["image_size"],
            )
        )

    def _on_success(self, success: bool) -> None:
        if success:
            self._append("[done] Conversion succeeded.")
        else:
            self._append("[failed] Conversion failed.")

    def _on_error(self, message: str) -> None:
        self._append(f"[error] {message}")
        QMessageBox.critical(self, "Conversion failed", message)
