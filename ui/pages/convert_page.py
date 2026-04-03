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
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.conversion.converters import convert_model, detect_model_framework
from ui.lib.qt_alert import qt_operation_error
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.recent_run_history import append_recent_run
from mb.utils.translations import _
from ui.lib.task_progress import attach_progress_dialog
from ui.task_context import LongTaskContext
from ui.task_runner import start_task
from ui.lib.form_layout_i18n import apply_qform_label_column


class ConvertPage(QWidget):
    """UI scaffold for model conversion settings."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._head = QLabel()
        root.addWidget(self._head)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        group = QGroupBox("mb convert")
        form = QFormLayout(group)
        self._main_form = form
        self.input_model = QLineEdit()
        self.output_model = QLineEdit()
        self.framework = QComboBox()
        self.framework.addItems(["auto-detect", "pytorch", "keras"])
        self.target = QComboBox()
        self.target.addItems(["onnx", "safetensors"])
        self.architecture = QLineEdit()
        self.num_classes = QSpinBox()
        self.num_classes.setRange(0, 1_000_000)
        self.num_classes.setSpecialValueText(_("Required only for PyTorch -> ONNX"))
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 4096)
        self.image_size.setValue(224)

        form.addRow(_("Input model"), self._path_row(self.input_model, select_dir=False, save=False))
        form.addRow(_("Output model"), self._path_row(self.output_model, select_dir=False, save=True))
        form.addRow(_("Source framework"), self.framework)
        form.addRow(_("Target format"), self.target)
        form.addRow(_("Architecture"), self.architecture)
        form.addRow(_("Num classes"), self.num_classes)
        form.addRow(_("Image size"), self.image_size)
        root.addWidget(group)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton(_("Validate Conversion"))
        self.btn_convert = QPushButton(_("Run Conversion"))
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_convert)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText(
            _("Conversion validation and execution messages will appear here.")
        )
        root.addWidget(self.output, 1)

        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_convert.clicked.connect(self._run_conversion)

        self.retranslate_ui(refresh_output=False)

    def retranslate_ui(self, *, refresh_output: bool = True) -> None:
        self._head.setText(f"<h2>{_('Convert')}</h2>")
        self._intro.setText(_("Prepare conversion jobs for ONNX and SafeTensors targets."))
        apply_qform_label_column(
            self._main_form,
            [
                _("Input model"),
                _("Output model"),
                _("Source framework"),
                _("Target format"),
                _("Architecture"),
                _("Num classes"),
                _("Image size"),
            ],
        )
        self.num_classes.setSpecialValueText(_("Required only for PyTorch -> ONNX"))
        self._hint.setText(_("Note: architecture/num_classes are needed for PyTorch -> ONNX."))
        self.btn_validate.setText(_("Validate Conversion"))
        self.btn_convert.setText(_("Run Conversion"))
        self.output.setPlaceholderText(
            _("Conversion validation and execution messages will appear here.")
        )
        for edit in (self.input_model, self.output_model):
            row = edit.parentWidget()
            if row is not None:
                btn = row.findChild(QPushButton)
                if btn is not None:
                    btn.setText(_("Browse..."))
        if refresh_output:
            self._run_startup_validation()

    def _run_startup_validation(self) -> None:
        """Called from :meth:`MainWindow._run_page_startup_validation` after cache restore."""
        self.output.clear()
        self._validate_inputs()

    def collect_gui_state(self) -> dict:
        """Serializable form state; restored by :class:`ui.controllers.cache_controller.CacheController`."""
        return {
            "input_model": self.input_model.text(),
            "output_model": self.output_model.text(),
            "framework_idx": int(self.framework.currentIndex()),
            "target_idx": int(self.target.currentIndex()),
            "architecture": self.architecture.text(),
            "num_classes": int(self.num_classes.value()),
            "image_size": int(self.image_size.value()),
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        try:
            self.input_model.setText(str(state.get("input_model", "")))
            self.output_model.setText(str(state.get("output_model", "")))
            fi = state.get("framework_idx")
            if isinstance(fi, int) and 0 <= fi < self.framework.count():
                self.framework.setCurrentIndex(fi)
            ti = state.get("target_idx")
            if isinstance(ti, int) and 0 <= ti < self.target.count():
                self.target.setCurrentIndex(ti)
            self.architecture.setText(str(state.get("architecture", "")))
            nc = state.get("num_classes")
            if isinstance(nc, int):
                self.num_classes.setValue(nc)
            iz = state.get("image_size")
            if isinstance(iz, int):
                self.image_size.setValue(iz)
        except Exception:
            pass

    def _path_row(self, edit: QLineEdit, select_dir: bool = True, save: bool = False) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton(_("Browse..."))
        browse.clicked.connect(lambda: self._browse(edit, select_dir=select_dir, save=save))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, select_dir: bool = True, save: bool = False) -> None:
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
        elif save:
            value = QFileDialog.getSaveFileName(
                self,
                _("Select output file"),
                start,
                _("Model files (*.onnx *.safetensors);;All files (*.*)"),
                options=QFileDialog.Option.DontUseNativeDialog,
            )[0]
            if value:
                edit.setText(value)
        else:
            value = QFileDialog.getOpenFileName(
                self,
                _("Select model file"),
                start,
                _("Model files (*.pth *.pt *.h5 *.keras *.onnx *.safetensors);;All files (*.*)"),
                options=QFileDialog.Option.DontUseNativeDialog,
            )[0]
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
            raise ValueError(_("Input model file does not exist."))
        if not output_path.parent.exists():
            raise ValueError(_("Output model parent directory does not exist."))

        framework = self.framework.currentText()
        source_framework = None if framework == "auto-detect" else framework
        target = self.target.currentText()
        architecture = self.architecture.text().strip() or None
        num_classes = int(self.num_classes.value()) if self.num_classes.value() > 0 else None

        detected = source_framework or detect_model_framework(input_path)
        if detected == "pytorch" and target == "onnx":
            if architecture is None or num_classes is None:
                raise ValueError(_("PyTorch -> ONNX requires architecture and num classes."))

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
            self._append(
                _("[ok] conversion inputs valid (source={src})").format(
                    src=detected or _("unknown")
                )
            )
        except ValueError as exc:
            self.btn_convert.setEnabled(False)
            self.btn_convert.setToolTip(str(exc))
            self._append(_("[invalid] {err}").format(err=exc))

    def _run_conversion(self) -> None:
        payload = self._collect_inputs()
        self._append(
            f"[run] mb convert {payload['input_path'].name} -> {payload['target_format']}"
        )
        self._pending_convert_summary = f"{payload['input_path'].name} → {payload['target_format']}"
        self._set_busy(True)
        handle = start_task(
            self._execute_conversion,
            self._on_success,
            self._on_error,
            lambda: self._set_busy(False),
            payload,
            pass_context=True,
            on_cancelled=self._on_convert_cancelled,
        )
        attach_progress_dialog(self, _("Convert model"), handle, cancellable=True)

    def _execute_conversion(self, ctx: LongTaskContext, payload: dict) -> bool:
        return bool(
            convert_model(
                input_path=payload["input_path"],
                output_path=payload["output_path"],
                source_framework=payload["source_framework"],
                target_format=payload["target_format"],
                architecture=payload["architecture"],
                num_classes=payload["num_classes"],
                image_size=payload["image_size"],
                cancel_event=ctx.cancel_event,
            )
        )

    def _on_convert_cancelled(self) -> None:
        self._append(_("[stopped] Conversion cancelled before completion."))
        append_recent_run(
            ModelBuilderTaskType.CONVERT,
            getattr(self, "_pending_convert_summary", "mb convert"),
            False,
            "cancelled",
        )

    def _on_success(self, success: bool) -> None:
        summary = getattr(self, "_pending_convert_summary", "mb convert")
        if success:
            self._append(_("[done] Conversion succeeded."))
            append_recent_run(ModelBuilderTaskType.CONVERT, summary, True)
        else:
            self._append(_("[failed] Conversion failed."))
            append_recent_run(ModelBuilderTaskType.CONVERT, summary, False, "reported failure")

    def _on_error(self, message: str) -> None:
        self._append(_("[error] {err}").format(err=message))
        append_recent_run(
            ModelBuilderTaskType.CONVERT,
            getattr(self, "_pending_convert_summary", "mb convert"),
            False,
            message,
        )
        qt_operation_error(
            self,
            _("Conversion failed"),
            _("Model conversion reported an error. See Details for the underlying message."),
            detail=message,
        )
