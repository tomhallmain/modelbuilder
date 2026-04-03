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
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.conversion.converters import detect_model_framework
from ui.lib.qt_alert import qt_alert
from mb.models.frameworks.registry import list_architectures
from mb.utils.translations import _
from ui.lib.form_layout_i18n import apply_qform_label_column


class InfoPage(QWidget):
    """UI scaffold for `mb info model` and `mb info dataset`."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._head = QLabel()
        root.addWidget(self._head)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_model_tab(), "")
        self._tabs.addTab(self._build_dataset_tab(), "")
        root.addWidget(self._tabs, 1)

        self.btn_model_info.clicked.connect(self._inspect_model)
        self.btn_dataset_info.clicked.connect(self._inspect_dataset)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self._head.setText(f"<h2>{_('Info')}</h2>")
        self._intro.setText(_("Inspect model metadata and dataset structure/statistics."))
        self._tabs.setTabText(0, _("Model"))
        self._tabs.setTabText(1, _("Dataset"))
        apply_qform_label_column(self._model_form, [_("Model path")])
        apply_qform_label_column(self._dataset_form, [_("Data dir")])
        self.btn_model_info.setText(_("Inspect Model"))
        self.btn_dataset_info.setText(_("Inspect Dataset"))
        self.model_output.setPlaceholderText(_("Model info output will appear here."))
        self.dataset_output.setPlaceholderText(_("Dataset info output will appear here."))
        for edit in (self.model_path, self.dataset_dir):
            row = edit.parentWidget()
            if row is not None:
                btn = row.findChild(QPushButton)
                if btn is not None:
                    btn.setText(_("Browse..."))

    def collect_gui_state(self) -> dict:
        """Serializable form state; restored by :class:`ui.controllers.cache_controller.CacheController`."""
        return {
            "tab": int(self._tabs.currentIndex()),
            "model_path": self.model_path.text(),
            "dataset_dir": self.dataset_dir.text(),
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        try:
            t = state.get("tab")
            if isinstance(t, int) and 0 <= t < self._tabs.count():
                self._tabs.setCurrentIndex(t)
            self.model_path.setText(str(state.get("model_path", "")))
            self.dataset_dir.setText(str(state.get("dataset_dir", "")))
        except Exception:
            pass

    def _build_model_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb info model")
        form = QFormLayout(group)
        self._model_form = form
        self.model_path = QLineEdit()
        form.addRow(_("Model path"), self._path_row(self.model_path, select_dir=False))
        v.addWidget(group)

        actions = QHBoxLayout()
        self.btn_model_info = QPushButton(_("Inspect Model"))
        actions.addWidget(self.btn_model_info)
        actions.addStretch(1)
        v.addLayout(actions)

        self.model_output = QTextEdit()
        self.model_output.setReadOnly(True)
        self.model_output.setPlaceholderText(_("Model info output will appear here."))
        v.addWidget(self.model_output, 1)
        return tab

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        group = QGroupBox("mb info dataset")
        form = QFormLayout(group)
        self._dataset_form = form
        self.dataset_dir = QLineEdit("data")
        form.addRow(_("Data dir"), self._path_row(self.dataset_dir, select_dir=True))
        v.addWidget(group)

        actions = QHBoxLayout()
        self.btn_dataset_info = QPushButton(_("Inspect Dataset"))
        actions.addWidget(self.btn_dataset_info)
        actions.addStretch(1)
        v.addLayout(actions)

        self.dataset_output = QTextEdit()
        self.dataset_output.setReadOnly(True)
        self.dataset_output.setPlaceholderText(_("Dataset info output will appear here."))
        v.addWidget(self.dataset_output, 1)

        return tab

    def _path_row(self, edit: QLineEdit, select_dir: bool = True) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton(_("Browse..."))
        browse.clicked.connect(lambda: self._browse(edit, select_dir=select_dir))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, select_dir: bool = True) -> None:
        start = edit.text().strip() or str(Path.cwd())
        if select_dir:
            value = QFileDialog.getExistingDirectory(
                self,
                _("Select directory"),
                start,
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)
        else:
            value, _ = QFileDialog.getOpenFileName(
                self,
                _("Select model file"),
                start,
                _("Model files (*.pth *.pt *.h5 *.keras *.onnx *.safetensors);;All files (*.*)"),
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if value:
                edit.setText(value)

    def _inspect_model(self) -> None:
        path_text = self.model_path.text().strip()
        if not path_text:
            qt_alert(
                self,
                _("Model path required"),
                _("Please provide a model path."),
                kind="warning",
            )
            return
        model_path = Path(path_text)
        if not model_path.exists():
            qt_alert(
                self,
                _("Missing file"),
                _("Model path not found: {path}").format(path=model_path),
                kind="warning",
            )
            return

        framework = detect_model_framework(model_path)
        lines = [
            _("Path: {path}").format(path=model_path),
            _("Size: {n:,} bytes").format(n=model_path.stat().st_size),
            _("Detected framework/type: {fw}").format(fw=framework or _("unknown")),
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
        lines.append(_("Registered architectures:"))
        for fw, items in archs.items():
            lines.append(
                "- {fw}: {names}".format(
                    fw=fw,
                    names=", ".join(items) if items else _("(none)"),
                )
            )

        self.model_output.setPlainText("\n".join(lines))

    def _inspect_dataset(self) -> None:
        data_dir = Path(self.dataset_dir.text().strip() or "data")
        if not data_dir.exists():
            qt_alert(
                self,
                _("Missing directory"),
                _("Data directory not found: {path}").format(path=data_dir),
                kind="warning",
            )
            return

        lines = [_("Data dir: {path}").format(path=data_dir), ""]
        for split in ["train", "test"]:
            split_dir = data_dir / split
            lines.append(f"[{split}]")
            if not split_dir.exists():
                lines.append(_("  missing"))
                lines.append("")
                continue
            class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
            total = 0
            for cls in class_dirs:
                count = len([x for x in cls.glob("*.jpg")]) + len([x for x in cls.glob("*.jpeg")]) + len([x for x in cls.glob("*.png")])
                total += count
                lines.append(f"  {cls.name}: {count}")
            lines.append(_("  total: {n}").format(n=total))
            lines.append("")
        self.dataset_output.setPlainText("\n".join(lines))
