"""Model bundle export page scaffold aligned with `mb export bundle`."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
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

from mb.export.bundle import export_bundle
from mb.pipeline_config import get_pipeline_config
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.recent_run_history import append_recent_run
from mb.utils.snapshot import run_id_from_latest_unified_snapshot
from mb.utils.translations import _
from ui.lib.fast_directory_picker_qt import get_existing_directory, get_open_file_name
from ui.lib.form_layout_i18n import apply_qform_label_column
from ui.lib.qt_alert import qt_operation_error
from ui.lib.task_progress import attach_progress_dialog
from ui.task_context import LongTaskContext
from ui.task_runner import start_task


class ExportPage(QWidget):
    """UI scaffold for model bundle export settings."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._head = QLabel()
        root.addWidget(self._head)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        group = QGroupBox("mb export bundle")
        form = QFormLayout(group)
        self._main_form = form

        self.input_model = QLineEdit()
        self.architecture = QLineEdit()
        self.num_classes = QSpinBox()
        self.num_classes.setRange(0, 1_000_000)
        self.num_classes.setSpecialValueText(_("Auto"))
        self.data_dir = QLineEdit()
        self.image_size = QSpinBox()
        self.image_size.setRange(0, 4096)
        self.image_size.setSpecialValueText(_("Auto"))
        self.image_size.setValue(0)
        self.run_id = QLineEdit()
        self.emit_arch_py = QCheckBox()
        self.emit_arch_py.setChecked(True)

        form.addRow(_("Input model"), self._path_row(self.input_model, select_file=True))
        form.addRow(_("Architecture"), self.architecture)
        form.addRow(_("Num classes"), self.num_classes)
        form.addRow(_("Dataset directory"), self._path_row(self.data_dir, select_file=False))
        form.addRow(_("Image size"), self.image_size)
        form.addRow(_("Run ID (optional)"), self.run_id)
        form.addRow(_("Generate architecture stub"), self.emit_arch_py)
        root.addWidget(group)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton(_("Validate Export"))
        self.btn_export = QPushButton(_("Run Export"))
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_export)
        actions.addStretch(1)
        root.addLayout(actions)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        root.addWidget(self.output, 1)

        self.btn_validate.clicked.connect(self._validate_inputs)
        self.btn_export.clicked.connect(self._run_export)
        self._prefill_from_pipeline()

        self.retranslate_ui(refresh_output=False)

    def retranslate_ui(self, *, refresh_output: bool = True) -> None:
        self._head.setText(f"<h2>{_('Export')}</h2>")
        self._intro.setText(_("Export a deployment bundle (SafeTensors + manifest) after conversion/training."))
        apply_qform_label_column(
            self._main_form,
            [
                _("Input model"),
                _("Architecture"),
                _("Num classes"),
                _("Dataset directory"),
                _("Image size"),
                _("Run ID (optional)"),
                _("Generate architecture stub"),
            ],
        )
        self.num_classes.setSpecialValueText(_("Auto"))
        self.image_size.setSpecialValueText(_("Auto"))
        self._hint.setText(
            _(
                "Class names are auto-resolved from snapshot or dataset directory. "
                "Tip: leave architecture/num classes/image size on Auto to use defaults."
            )
        )
        self.btn_validate.setText(_("Validate Export"))
        self.btn_export.setText(_("Run Export"))
        self.output.setPlaceholderText(_("Export validation and execution messages will appear here."))
        for edit in (self.input_model, self.data_dir):
            row = edit.parentWidget()
            if row is not None:
                btn = row.findChild(QPushButton)
                if btn is not None:
                    btn.setText(_("Browse..."))
        if refresh_output:
            self._run_startup_validation()

    def _run_startup_validation(self) -> None:
        self.output.clear()
        self._prefill_from_pipeline()
        self._sync_input_model_from_convert_page_if_empty()
        self._validate_inputs()

    def collect_gui_state(self) -> dict:
        return {
            "input_model": self.input_model.text(),
            "architecture": self.architecture.text(),
            "num_classes": int(self.num_classes.value()),
            "data_dir": self.data_dir.text(),
            "image_size": int(self.image_size.value()),
            "run_id": self.run_id.text(),
            "emit_arch_py": bool(self.emit_arch_py.isChecked()),
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        # Keep pipeline-prefilled defaults when cache contains empty strings.
        input_model = str(state.get("input_model", "")).strip()
        if input_model:
            self.input_model.setText(input_model)
        architecture = str(state.get("architecture", "")).strip()
        if architecture:
            self.architecture.setText(architecture)
        nc = state.get("num_classes")
        if isinstance(nc, int):
            self.num_classes.setValue(nc)
        data_dir = str(state.get("data_dir", "")).strip()
        if data_dir:
            self.data_dir.setText(data_dir)
        iz = state.get("image_size")
        if isinstance(iz, int):
            self.image_size.setValue(iz)
        run_id = str(state.get("run_id", "")).strip()
        if run_id:
            self.run_id.setText(run_id)
        self.emit_arch_py.setChecked(bool(state.get("emit_arch_py", True)))
        self._prefill_from_pipeline()
        self._sync_input_model_from_convert_page_if_empty()

    def _path_row(self, edit: QLineEdit, *, select_file: bool) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton(_("Browse..."))
        browse.clicked.connect(lambda: self._browse(edit, select_file=select_file))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, *, select_file: bool) -> None:
        start = edit.text().strip() or str(Path.cwd())
        if select_file:
            value = get_open_file_name(
                self,
                _("Select model file"),
                start,
                _("Model files (*.pth *.pt);;All files (*.*)"),
            )
        else:
            value = get_existing_directory(self, _("Select directory"), start)
        if value:
            edit.setText(value)
        self._validate_inputs()

    def _append(self, msg: str) -> None:
        self.output.append(msg)

    def _set_busy(self, busy: bool) -> None:
        self.btn_validate.setEnabled(not busy)
        self.btn_export.setEnabled(not busy and self._can_run())

    def _collect_inputs(self) -> dict:
        input_path = Path(self.input_model.text().strip())
        if not input_path.exists():
            raise ValueError(_("Input model file does not exist."))
        output_dir = input_path.parent
        if not output_dir.exists():
            raise ValueError(_("Input model parent directory does not exist."))

        data_dir_text = self.data_dir.text().strip()

        return {
            "input_model": input_path,
            "output_dir": output_dir,
            "architecture": self.architecture.text().strip() or None,
            "num_classes": int(self.num_classes.value()) if self.num_classes.value() > 0 else None,
            "class_names": None,
            "data_dir": Path(data_dir_text) if data_dir_text else None,
            "image_size": int(self.image_size.value()) if self.image_size.value() > 0 else None,
            "include_architecture_py": bool(self.emit_arch_py.isChecked()),
            "run_id": self.run_id.text().strip() or None,
            "snapshot_path": None,
            "pipeline_config": get_pipeline_config().to_dict(),
        }

    def _prefill_from_pipeline(self) -> None:
        pc = get_pipeline_config()
        arch = pc.get("model.default_architecture")
        if isinstance(arch, str) and arch.strip() and not self.architecture.text().strip():
            self.architecture.setText(arch.strip())
        data_dir = pc.get("data.data_dir")
        if isinstance(data_dir, str) and data_dir.strip() and not self.data_dir.text().strip():
            self.data_dir.setText(data_dir.strip())
        image_size = pc.get("data.image_size")
        try:
            parsed = int(float(image_size))
            if parsed > 0:
                self.image_size.setValue(parsed)
        except Exception:
            pass
        if not self.run_id.text().strip():
            raw_dir = pc.get("data.raw_data_dir")
            search_paths: list[Path] = []
            if isinstance(data_dir, str) and data_dir.strip():
                search_paths.append(Path(data_dir.strip()))
            if isinstance(raw_dir, str) and raw_dir.strip():
                search_paths.append(Path(raw_dir.strip()))
            if search_paths:
                rid = run_id_from_latest_unified_snapshot(search_paths, quiet=True)
                if isinstance(rid, str) and rid.strip():
                    self.run_id.setText(rid.strip())

    def _sync_input_model_from_convert_page_if_empty(self) -> None:
        if self.input_model.text().strip():
            return
        w = self.window()
        page_widgets = getattr(w, "page_widgets", None)
        if not isinstance(page_widgets, list):
            return
        for page in page_widgets:
            convert_input = getattr(page, "input_model", None)
            if convert_input is None:
                continue
            text_fn = getattr(convert_input, "text", None)
            if not callable(text_fn):
                continue
            model_path = str(text_fn()).strip()
            if model_path:
                self.input_model.setText(model_path)
                break

    def showEvent(self, event) -> None:
        self._prefill_from_pipeline()
        self._sync_input_model_from_convert_page_if_empty()
        super().showEvent(event)

    def _can_run(self) -> bool:
        try:
            self._collect_inputs()
            return True
        except ValueError:
            return False

    def _validate_inputs(self) -> None:
        try:
            payload = self._collect_inputs()
            self.btn_export.setEnabled(True)
            self._append(
                _("[ok] export inputs valid (arch={arch}, classes={n})").format(
                    arch=payload.get("architecture") or _("auto"),
                    n=payload.get("num_classes") or _("auto"),
                )
            )
        except ValueError as exc:
            self.btn_export.setEnabled(False)
            self.btn_export.setToolTip(str(exc))
            self._append(_("[invalid] {err}").format(err=exc))

    def _run_export(self) -> None:
        payload = self._collect_inputs()
        self._pending_export_summary = f"{payload['input_model'].name} -> bundle"
        self._append(_("[run] mb export bundle"))
        self._set_busy(True)
        handle = start_task(
            self._execute_export,
            self._on_success,
            self._on_error,
            lambda: self._set_busy(False),
            payload,
            pass_context=True,
            on_cancelled=self._on_export_cancelled,
        )
        attach_progress_dialog(self, _("Export bundle"), handle, cancellable=True)

    def _execute_export(self, ctx: LongTaskContext, payload: dict) -> dict:
        if ctx.cancel_event.is_set():
            raise RuntimeError(_("Cancelled before export started"))
        return export_bundle(**payload)

    def _on_export_cancelled(self) -> None:
        self._append(_("[stopped] Export cancelled before completion."))
        append_recent_run(
            ModelBuilderTaskType.EXPORT,
            getattr(self, "_pending_export_summary", "mb export bundle"),
            False,
            "cancelled",
        )

    def _on_success(self, result: dict) -> None:
        self._append(_("[done] Export succeeded."))
        self._append(_("Weights: {path}").format(path=result.get("weights_path")))
        self._append(_("Manifest: {path}").format(path=result.get("manifest_path")))
        arch_path = result.get("architecture_path")
        if arch_path:
            self._append(_("Architecture stub: {path}").format(path=arch_path))
        else:
            self._append(_("[info] Architecture stub not generated (option disabled)."))
        append_recent_run(
            ModelBuilderTaskType.EXPORT,
            getattr(self, "_pending_export_summary", "mb export bundle"),
            True,
        )

    def _on_error(self, message: str) -> None:
        self._append(_("[error] {err}").format(err=message))
        append_recent_run(
            ModelBuilderTaskType.EXPORT,
            getattr(self, "_pending_export_summary", "mb export bundle"),
            False,
            message,
        )
        qt_operation_error(
            self,
            _("Export failed"),
            _("Model export reported an error. See Details for the underlying message."),
            detail=message,
        )

