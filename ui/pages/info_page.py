"""Information page scaffold for model/dataset queries."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.conversion.converters import detect_model_framework
from mb.models.frameworks.registry import list_architectures


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
        form.addRow("Model path", self._path_row(self.model_path, select_dir=False))
        v.addWidget(group)

        actions = QHBoxLayout()
        self.btn_model_info = QPushButton("Inspect Model")
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
        form.addRow("Data dir", self._path_row(self.dataset_dir, select_dir=True))
        v.addWidget(group)

        actions = QHBoxLayout()
        self.btn_dataset_info = QPushButton("Inspect Dataset")
        actions.addWidget(self.btn_dataset_info)
        actions.addStretch(1)
        v.addLayout(actions)

        self.dataset_output = QTextEdit()
        self.dataset_output.setReadOnly(True)
        self.dataset_output.setPlaceholderText("Dataset info output will appear here.")
        v.addWidget(self.dataset_output, 1)

        self.btn_model_info.clicked.connect(self._inspect_model)
        self.btn_dataset_info.clicked.connect(self._inspect_dataset)
        return tab

    def _path_row(self, edit: QLineEdit, select_dir: bool = True) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse(edit, select_dir=select_dir))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, select_dir: bool = True) -> None:
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

    def _inspect_model(self) -> None:
        path_text = self.model_path.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Model path required", "Please provide a model path.")
            return
        model_path = Path(path_text)
        if not model_path.exists():
            QMessageBox.warning(self, "Missing file", f"Model path not found: {model_path}")
            return

        framework = detect_model_framework(model_path)
        lines = [
            f"Path: {model_path}",
            f"Size: {model_path.stat().st_size:,} bytes",
            f"Detected framework/type: {framework or 'unknown'}",
            "",
        ]
        try:
            from mb.models.frameworks import pytorch  # noqa: F401
        except Exception:
            pass
        try:
            from mb.models.frameworks import keras  # noqa: F401
        except Exception:
            pass
        archs = list_architectures()
        lines.append("Registered architectures:")
        for fw, items in archs.items():
            lines.append(f"- {fw}: {', '.join(items) if items else '(none)'}")

        self.model_output.setPlainText("\n".join(lines))

    def _inspect_dataset(self) -> None:
        data_dir = Path(self.dataset_dir.text().strip() or "data")
        if not data_dir.exists():
            QMessageBox.warning(self, "Missing directory", f"Data directory not found: {data_dir}")
            return

        lines = [f"Data dir: {data_dir}", ""]
        for split in ["train", "test"]:
            split_dir = data_dir / split
            lines.append(f"[{split}]")
            if not split_dir.exists():
                lines.append("  missing")
                lines.append("")
                continue
            class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
            total = 0
            for cls in class_dirs:
                count = len([x for x in cls.glob("*.jpg")]) + len([x for x in cls.glob("*.jpeg")]) + len([x for x in cls.glob("*.png")])
                total += count
                lines.append(f"  {cls.name}: {count}")
            lines.append(f"  total: {total}")
            lines.append("")
        self.dataset_output.setPlainText("\n".join(lines))
