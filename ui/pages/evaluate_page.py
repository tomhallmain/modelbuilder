"""Evaluate page: metrics, misclassified, and compare (aligned with ``mb evaluate``)."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.evaluate.metrics import ClassificationMetricsReport, format_classification_report, run_evaluate_metrics
from mb.models.types import EvaluateSubcommand, FrameworkType, ModelType
from mb.pipeline_config import get_pipeline_config, reload_pipeline_config
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.recent_run_history import append_recent_run
from mb.utils.translations import _
from ui.lib.fast_directory_picker_qt import get_existing_directory, get_open_file_name
from ui.lib.form_layout_i18n import apply_qform_label_column
from ui.lib.qt_alert import qt_operation_error
from ui.lib.task_progress import attach_progress_dialog
from ui.task_context import LongTaskContext
from ui.task_runner import start_task


class EvaluatePage(QWidget):
    """
    Tabs for ``mb evaluate`` subcommands.

    Shared fields (data directory, model type, checkpoint, framework, architecture, …) match
    :class:`~ui.pages.train_page.TrainPage` / :class:`~ui.pages.export_page.ExportPage` conventions:
    pipeline YAML pre-fills empty paths on show, and we mirror the Train **data dir**,
    **Info**/**Export** dataset paths, and Convert **input model** when those fields are still empty
    so one workspace edit propagates.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("evaluate_page")
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._head = QLabel()
        self._head.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self._head)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        shared = QGroupBox()
        self._shared_group = shared
        sform = QFormLayout(shared)
        self._shared_form = sform

        self.data_dir = QLineEdit()
        self.data_dir.setObjectName("evaluate_data_dir_edit")
        self.model_type = QComboBox()
        for mt in ModelType:
            self.model_type.addItem(mt.value)
        self.model_path = QLineEdit()
        self.model_path.setObjectName("evaluate_model_path_edit")
        self.framework = QComboBox()
        self.framework.addItem(_("Infer from file"))
        self.framework.addItem(FrameworkType.PYTORCH.value)
        self.framework.addItem(FrameworkType.KERAS.value)
        self.architecture = QLineEdit()
        self.architecture.setObjectName("evaluate_architecture_edit")
        self.num_classes = QSpinBox()
        self.num_classes.setRange(0, 1_000_000)
        self.num_classes.setSpecialValueText(_("Auto"))
        self.num_classes.setValue(0)
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 4096)
        self.image_size.setValue(224)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 4096)
        self.batch_size.setValue(32)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 128)
        self.num_workers.setValue(0)
        self.device = QLineEdit()
        self.device.setPlaceholderText(_("optional, e.g. cuda or cpu"))

        sform.addRow(_("Eval data dir (ImageFolder)"), self._path_row(self.data_dir, is_file=False))
        sform.addRow(_("Model type"), self.model_type)
        sform.addRow(_("Model checkpoint"), self._path_row(self.model_path, is_file=True))
        sform.addRow(_("Framework"), self.framework)
        sform.addRow(_("Architecture"), self.architecture)
        sform.addRow(_("Num classes"), self.num_classes)
        sform.addRow(_("Image size"), self.image_size)
        sform.addRow(_("Batch size"), self.batch_size)
        sform.addRow(_("DataLoader workers"), self.num_workers)
        sform.addRow(_("Device"), self.device)
        root.addWidget(shared)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_metrics_tab(), "")
        self._tabs.addTab(self._build_misclassified_tab(), "")
        self._tabs.addTab(self._build_compare_tab(), "")
        root.addWidget(self._tabs, 0)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setObjectName("evaluate_output_log")
        self.output.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        root.addWidget(self.output, 1)

        self.metrics_validate.clicked.connect(lambda: self._run_cli(EvaluateSubcommand.METRICS, dry_run=True))
        self.metrics_run.clicked.connect(lambda: self._run_cli(EvaluateSubcommand.METRICS, dry_run=False))
        self.mis_validate.clicked.connect(lambda: self._run_cli(EvaluateSubcommand.MISCLASSIFIED, dry_run=True))
        self.mis_run.clicked.connect(lambda: self._run_cli(EvaluateSubcommand.MISCLASSIFIED, dry_run=False))
        self.cmp_validate.clicked.connect(lambda: self._run_cli(EvaluateSubcommand.COMPARE, dry_run=True))
        self.cmp_run.clicked.connect(lambda: self._run_cli(EvaluateSubcommand.COMPARE, dry_run=False))

        self.retranslate_ui(refresh_output=False)

    def retranslate_ui(self, *, refresh_output: bool = True) -> None:
        self._head.setText(f"<h2>{_('Evaluate')}</h2>")
        self._intro.setText(
            _(
                "Score checkpoints on a labeled split, list disagreements with folder labels, "
                "or compare two models in one pass (CLI: {cmd})."
            ).format(cmd="mb evaluate")
        )
        self._shared_group.setTitle(_("Shared with metrics / misclassified"))
        apply_qform_label_column(
            self._shared_form,
            [
                _("Eval data dir (ImageFolder)"),
                _("Model type"),
                _("Model checkpoint"),
                _("Framework"),
                _("Architecture"),
                _("Num classes"),
                _("Image size"),
                _("Batch size"),
                _("DataLoader workers"),
                _("Device"),
            ],
        )
        self.num_classes.setSpecialValueText(_("Auto"))
        self.framework.setItemText(0, _("Infer from file"))
        self.cmp_framework_b.setItemText(0, _("Same as primary"))
        self.device.setPlaceholderText(_("optional, e.g. cuda or cpu"))
        self._tabs.setTabText(0, _("Metrics"))
        self._tabs.setTabText(1, _("Misclassified"))
        self._tabs.setTabText(2, _("Compare"))
        self._metrics_group.setTitle(f"mb evaluate {EvaluateSubcommand.METRICS.value}")
        self._mis_group.setTitle(f"mb evaluate {EvaluateSubcommand.MISCLASSIFIED.value}")
        self._cmp_group.setTitle(f"mb evaluate {EvaluateSubcommand.COMPARE.value}")
        apply_qform_label_column(self._mis_form, [_("Max rows (0 = no cap)"), _("Output CSV")])
        apply_qform_label_column(
            self._cmp_form,
            [
                _("Model A"),
                _("Model B"),
                _("Framework B"),
                _("Architecture B"),
                _("Max disagreement rows (0 = no cap)"),
                _("Output TSV"),
            ],
        )
        self.mis_max_report.setSpecialValueText(_("No cap"))
        self.cmp_max_dis.setSpecialValueText(_("No cap"))
        self.metrics_validate.setText(_("Validate (dry-run)"))
        self.metrics_run.setText(_("Run metrics"))
        self.metrics_confusion_label.setText(_("Confusion matrix (rows = true class, cols = predicted)"))
        self.mis_validate.setText(_("Validate (dry-run)"))
        self.mis_run.setText(_("Run misclassified"))
        self.cmp_validate.setText(_("Validate (dry-run)"))
        self.cmp_run.setText(_("Run compare"))
        self.output.setPlaceholderText(_("Evaluation output will appear here."))
        for edit, is_file in (
            (self.data_dir, False),
            (self.model_path, True),
            (self.cmp_model_a, True),
            (self.cmp_model_b, True),
            (self.mis_output, False),
            (self.cmp_output, False),
        ):
            row = edit.parentWidget()
            if row is not None:
                btn = row.findChild(QPushButton)
                if btn is not None:
                    btn.setText(_("Browse..."))
        if refresh_output:
            self._run_startup_validation()

    def _run_startup_validation(self) -> None:
        self.output.clear()
        self._append(_("[info] Adjust paths and choose a tab, then Validate or Run."))

    def collect_gui_state(self) -> dict:
        return {
            "tab": int(self._tabs.currentIndex()),
            "data_dir": self.data_dir.text(),
            "model_type_idx": int(self.model_type.currentIndex()),
            "model_path": self.model_path.text(),
            "framework_idx": int(self.framework.currentIndex()),
            "architecture": self.architecture.text(),
            "num_classes": int(self.num_classes.value()),
            "image_size": int(self.image_size.value()),
            "batch_size": int(self.batch_size.value()),
            "num_workers": int(self.num_workers.value()),
            "device": self.device.text(),
            "mis_max_report": int(self.mis_max_report.value()),
            "mis_output": self.mis_output.text(),
            "cmp_model_a": self.cmp_model_a.text(),
            "cmp_model_b": self.cmp_model_b.text(),
            "cmp_framework_b_idx": int(self.cmp_framework_b.currentIndex()),
            "cmp_arch_b": self.cmp_arch_b.text(),
            "cmp_max_dis": int(self.cmp_max_dis.value()),
            "cmp_output": self.cmp_output.text(),
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        try:
            t = state.get("tab")
            if isinstance(t, int) and 0 <= t < self._tabs.count():
                self._tabs.setCurrentIndex(t)
            self.data_dir.setText(str(state.get("data_dir", "")))
            mti = state.get("model_type_idx")
            if isinstance(mti, int) and 0 <= mti < self.model_type.count():
                self.model_type.setCurrentIndex(mti)
            self.model_path.setText(str(state.get("model_path", "")))
            fi = state.get("framework_idx")
            if isinstance(fi, int) and 0 <= fi < self.framework.count():
                self.framework.setCurrentIndex(fi)
            self.architecture.setText(str(state.get("architecture", "")))
            nc = state.get("num_classes")
            if isinstance(nc, int):
                self.num_classes.setValue(nc)
            for key, spin in (
                ("image_size", self.image_size),
                ("batch_size", self.batch_size),
                ("num_workers", self.num_workers),
            ):
                v = state.get(key)
                if isinstance(v, int):
                    spin.setValue(v)
            self.device.setText(str(state.get("device", "")))
            mm = state.get("mis_max_report")
            if isinstance(mm, int):
                self.mis_max_report.setValue(mm)
            self.mis_output.setText(str(state.get("mis_output", "")))
            self.cmp_model_a.setText(str(state.get("cmp_model_a", "")))
            self.cmp_model_b.setText(str(state.get("cmp_model_b", "")))
            cfi = state.get("cmp_framework_b_idx")
            if isinstance(cfi, int) and 0 <= cfi < self.cmp_framework_b.count():
                self.cmp_framework_b.setCurrentIndex(cfi)
            self.cmp_arch_b.setText(str(state.get("cmp_arch_b", "")))
            cd = state.get("cmp_max_dis")
            if isinstance(cd, int):
                self.cmp_max_dis.setValue(cd)
            self.cmp_output.setText(str(state.get("cmp_output", "")))
        except Exception:
            pass
        self._prefill_from_pipeline()

    def showEvent(self, event) -> None:
        self._prefill_from_pipeline()
        self._sync_data_dir_from_train_or_info_if_empty()
        self._sync_model_from_convert_if_empty()
        self._sync_compare_models_from_shared_if_empty()
        super().showEvent(event)

    def _prefill_from_pipeline(self) -> None:
        pc = get_pipeline_config()
        dd = pc.get("data.data_dir")
        if isinstance(dd, str) and dd.strip() and not self.data_dir.text().strip():
            self.data_dir.setText(dd.strip())
        arch = pc.get("model.default_architecture")
        if isinstance(arch, str) and arch.strip() and not self.architecture.text().strip():
            self.architecture.setText(arch.strip())
        iz = pc.get("data.image_size")
        try:
            parsed = int(float(iz))
            if parsed > 0:
                self.image_size.setValue(parsed)
        except Exception:
            pass

    def _sync_data_dir_from_train_or_info_if_empty(self) -> None:
        if self.data_dir.text().strip():
            return
        w = self.window()
        pages = getattr(w, "page_widgets", None)
        if not isinstance(pages, list):
            return
        for p in pages:
            name = p.__class__.__name__
            if name == "TrainPage":
                td = getattr(p, "data_dir", None)
                if td is not None and callable(getattr(td, "text", None)):
                    t = str(td.text()).strip()
                    if t:
                        self.data_dir.setText(t)
                        return
            if name == "InfoPage":
                ds = getattr(p, "dataset_dir", None)
                if ds is not None and callable(getattr(ds, "text", None)):
                    t = str(ds.text()).strip()
                    if t:
                        self.data_dir.setText(t)
                        return
            if name == "ExportPage":
                ed = getattr(p, "data_dir", None)
                if ed is not None and callable(getattr(ed, "text", None)):
                    t = str(ed.text()).strip()
                    if t:
                        self.data_dir.setText(t)
                        return

    def _sync_model_from_convert_if_empty(self) -> None:
        if self.model_path.text().strip():
            return
        w = self.window()
        pages = getattr(w, "page_widgets", None)
        if not isinstance(pages, list):
            return
        for p in pages:
            conv = getattr(p, "input_model", None)
            if conv is None or not callable(getattr(conv, "text", None)):
                continue
            t = str(conv.text()).strip()
            if t:
                self.model_path.setText(t)
                return

    def _sync_compare_models_from_shared_if_empty(self) -> None:
        sm = self.model_path.text().strip()
        if not sm:
            return
        if not self.cmp_model_a.text().strip():
            self.cmp_model_a.setText(sm)
        if not self.cmp_model_b.text().strip():
            self.cmp_model_b.setText(sm)

    def _path_row(self, edit: QLineEdit, *, is_file: bool) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton(_("Browse..."))
        browse.clicked.connect(lambda: self._browse(edit, is_file=is_file))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, *, is_file: bool) -> None:
        start = edit.text().strip() or str(Path.cwd())
        if is_file:
            path = get_open_file_name(
                self,
                _("Select file"),
                start,
                _("Models (*.pth *.pt *.h5 *.keras);;All files (*.*)"),
            )
        else:
            path = get_existing_directory(self, _("Select directory"), start)
        if path:
            edit.setText(path)

    def _build_metrics_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        self._metrics_group = QGroupBox()
        v.addWidget(self._metrics_group)
        row = QHBoxLayout()
        self.metrics_validate = QPushButton()
        self.metrics_run = QPushButton()
        row.addWidget(self.metrics_validate)
        row.addWidget(self.metrics_run)
        row.addStretch(1)
        v.addLayout(row)

        self.metrics_confusion_label = QLabel()
        self.metrics_confusion_label.setWordWrap(True)
        v.addWidget(self.metrics_confusion_label)
        self.metrics_confusion_table = QTableWidget()
        self.metrics_confusion_table.setObjectName("evaluate_metrics_confusion_table")
        self.metrics_confusion_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_confusion_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        v.addWidget(self.metrics_confusion_table, 1)
        self._populate_confusion_matrix(None)

        v.addStretch(1)
        return tab

    def _build_misclassified_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        self._mis_group = QGroupBox()
        mform = QFormLayout(self._mis_group)
        self._mis_form = mform
        self.mis_max_report = QSpinBox()
        self.mis_max_report.setRange(0, 10_000_000)
        self.mis_max_report.setSpecialValueText(_("No cap"))
        self.mis_max_report.setValue(0)
        self.mis_output = QLineEdit()
        mform.addRow(_("Max rows (0 = no cap)"), self.mis_max_report)
        mform.addRow(_("Output CSV"), self._path_row(self.mis_output, is_file=False))
        v.addWidget(self._mis_group)
        row = QHBoxLayout()
        self.mis_validate = QPushButton()
        self.mis_run = QPushButton()
        row.addWidget(self.mis_validate)
        row.addWidget(self.mis_run)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        return tab

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        self._cmp_group = QGroupBox()
        cform = QFormLayout(self._cmp_group)
        self._cmp_form = cform
        self.cmp_model_a = QLineEdit()
        self.cmp_model_b = QLineEdit()
        self.cmp_framework_b = QComboBox()
        self.cmp_framework_b.addItem(_("Same as primary"))
        self.cmp_framework_b.addItem(FrameworkType.PYTORCH.value)
        self.cmp_framework_b.addItem(FrameworkType.KERAS.value)
        self.cmp_arch_b = QLineEdit()
        self.cmp_max_dis = QSpinBox()
        self.cmp_max_dis.setRange(0, 10_000_000)
        self.cmp_max_dis.setSpecialValueText(_("No cap"))
        self.cmp_max_dis.setValue(0)
        self.cmp_output = QLineEdit()
        cform.addRow(_("Model A"), self._path_row(self.cmp_model_a, is_file=True))
        cform.addRow(_("Model B"), self._path_row(self.cmp_model_b, is_file=True))
        cform.addRow(_("Framework B"), self.cmp_framework_b)
        cform.addRow(_("Architecture B"), self.cmp_arch_b)
        cform.addRow(_("Max disagreement rows (0 = no cap)"), self.cmp_max_dis)
        cform.addRow(_("Output TSV"), self._path_row(self.cmp_output, is_file=False))
        v.addWidget(self._cmp_group)
        hint = QLabel()
        hint.setWordWrap(True)
        hint.setText(
            _(
                "Compare uses the same eval directory, image size, batch, workers, and device as above. "
                "Model A uses the shared architecture; set architecture B if checkpoint B differs."
            )
        )
        v.addWidget(hint)
        row = QHBoxLayout()
        self.cmp_validate = QPushButton()
        self.cmp_run = QPushButton()
        row.addWidget(self.cmp_validate)
        row.addWidget(self.cmp_run)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        return tab

    def _config_argv_prefix(self) -> list[str]:
        w = self.window()
        fn = getattr(w, "_effective_pipeline_config_path", None)
        path = fn() if callable(fn) else None
        if path:
            return ["--config", str(Path(path).resolve())]
        return []

    def _append(self, msg: str) -> None:
        self.output.append(msg)

    def _shared_argv_tail(self) -> list[str]:
        tail: list[str] = [
            "--data-dir",
            self.data_dir.text().strip(),
            "--model-type",
            self.model_type.currentText().strip() or ModelType.IMAGE_CLASSIFICATION.value,
            "--image-size",
            str(int(self.image_size.value())),
            "--batch-size",
            str(int(self.batch_size.value())),
            "--num-workers",
            str(int(self.num_workers.value())),
        ]
        fwi = int(self.framework.currentIndex())
        if fwi == 1:
            tail.extend(["--framework", FrameworkType.PYTORCH.value])
        elif fwi == 2:
            tail.extend(["--framework", FrameworkType.KERAS.value])
        arch = self.architecture.text().strip()
        if arch:
            tail.extend(["--architecture", arch])
        if int(self.num_classes.value()) > 0:
            tail.extend(["--num-classes", str(int(self.num_classes.value()))])
        dev = self.device.text().strip()
        if dev:
            tail.extend(["--device", dev])
        return tail

    def _argv_for(self, sub: EvaluateSubcommand, *, dry_run: bool) -> list[str]:
        base = self._config_argv_prefix() + [
            ModelBuilderTaskType.EVALUATE.value,
            sub.value,
        ]
        if sub in (EvaluateSubcommand.METRICS, EvaluateSubcommand.MISCLASSIFIED):
            base.extend(
                [
                    "--model",
                    self.model_path.text().strip(),
                    *self._shared_argv_tail(),
                ]
            )
        else:
            base.extend(
                [
                    "--model-a",
                    self.cmp_model_a.text().strip(),
                    "--model-b",
                    self.cmp_model_b.text().strip(),
                    *self._shared_argv_tail(),
                ]
            )
            bfwi = int(self.cmp_framework_b.currentIndex())
            if bfwi == 1:
                base.extend(["--framework-b", FrameworkType.PYTORCH.value])
            elif bfwi == 2:
                base.extend(["--framework-b", FrameworkType.KERAS.value])
            ab = self.cmp_arch_b.text().strip()
            if ab:
                base.extend(["--architecture-b", ab])
            if int(self.cmp_max_dis.value()) > 0:
                base.extend(["--max-disagreement-report", str(int(self.cmp_max_dis.value()))])
            out = self.cmp_output.text().strip()
            if out:
                base.extend(["--output", out])

        if sub == EvaluateSubcommand.MISCLASSIFIED:
            if int(self.mis_max_report.value()) > 0:
                base.extend(["--max-report", str(int(self.mis_max_report.value()))])
            mo = self.mis_output.text().strip()
            if mo:
                base.extend(["--output", mo])

        if dry_run:
            base.append("--dry-run")
        return base

    def _set_busy(self, busy: bool) -> None:
        for b in (
            self.metrics_validate,
            self.metrics_run,
            self.mis_validate,
            self.mis_run,
            self.cmp_validate,
            self.cmp_run,
        ):
            b.setEnabled(not busy)

    def _run_cli(self, sub: EvaluateSubcommand, *, dry_run: bool) -> None:
        try:
            argv = self._argv_for(sub, dry_run=dry_run)
        except Exception as exc:
            qt_operation_error(self, _("Evaluate"), str(exc), detail=str(exc))
            return
        label = f"mb evaluate {sub.value}" + (" --dry-run" if dry_run else "")
        self._append(_("[run] {cmd}").format(cmd=" ".join(argv)))
        self._pending_eval_summary = label
        self._set_busy(True)
        # Stale from a previous real metrics run; cleared up front so a dry-run or a
        # misclassified/compare run in between doesn't leave an unrelated matrix on screen.
        self._populate_confusion_matrix(None)
        if sub == EvaluateSubcommand.METRICS and not dry_run:
            handle = start_task(
                self._worker_metrics_report_main,
                self._on_metrics_report_success,
                self._on_eval_error,
                lambda: self._set_busy(False),
                argv,
                pass_context=True,
            )
        else:
            handle = start_task(
                self._worker_evaluate_main,
                self._on_eval_success,
                self._on_eval_error,
                lambda: self._set_busy(False),
                argv,
                pass_context=True,
            )
        attach_progress_dialog(self, _("Evaluate"), handle, cancellable=False)

    def _worker_evaluate_main(self, ctx: LongTaskContext, argv: list[str]) -> tuple[int, str, str]:
        ctx.progress(_("Running…"), None, True)
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        from mb.cli import main

        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = main(argv)
        return int(code), out_buf.getvalue(), err_buf.getvalue()

    def _worker_metrics_report_main(
        self, ctx: LongTaskContext, argv: list[str]
    ) -> Tuple[int, str, Optional[ClassificationMetricsReport]]:
        """
        Real (non-dry-run) ``metrics`` runs go through :func:`run_evaluate_metrics` directly
        (same parser + pipeline-config reload as ``mb.cli.main``) instead of ``main(argv)``, so
        the confusion matrix viewer gets the structured report without a second inference pass
        or scraping it back out of CLI stdout text.
        """
        ctx.progress(_("Running…"), None, True)
        from mb.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(argv)
        reload_pipeline_config(getattr(parsed, "config", None), force=True)
        code, report = run_evaluate_metrics(parsed)
        text = format_classification_report(report) if report is not None else ""
        return int(code), text, report

    def _on_metrics_report_success(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 3:
            self._append(_("[done]"))
            return
        code, text, report = payload
        if text.strip():
            self._append(text.rstrip())
        ok = code == 0
        self._populate_confusion_matrix(report if ok else None)
        if ok:
            self._append(_("[done] Exit code 0."))
            append_recent_run(
                ModelBuilderTaskType.EVALUATE,
                getattr(self, "_pending_eval_summary", "mb evaluate"),
                True,
            )
        else:
            self._append(_("[failed] Exit code {c}.").format(c=code))
            append_recent_run(
                ModelBuilderTaskType.EVALUATE,
                getattr(self, "_pending_eval_summary", "mb evaluate"),
                False,
                f"exit {code}",
            )

    def _populate_confusion_matrix(self, report: Optional[ClassificationMetricsReport]) -> None:
        table = self.metrics_confusion_table
        if report is None or not report.confusion_matrix:
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        names = report.class_names
        cm = report.confusion_matrix
        table.setRowCount(len(names))
        table.setColumnCount(len(names))
        table.setHorizontalHeaderLabels(names)
        table.setVerticalHeaderLabels(names)
        max_val = max((v for row in cm for v in row), default=0)
        for i, row in enumerate(cm):
            for j, v in enumerate(row):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if max_val > 0:
                    intensity = int(255 * (v / max_val))
                    if i == j:
                        item.setBackground(QColor(255 - intensity, 255, 255 - intensity))
                    else:
                        item.setBackground(QColor(255, 255 - intensity, 255 - intensity))
                table.setItem(i, j, item)
        table.resizeColumnsToContents()

    def _on_eval_success(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 3:
            self._append(_("[done]"))
            return
        code, out, err = payload
        if out.strip():
            self._append(out.rstrip())
        if err.strip():
            self._append(err.rstrip())
        ok = code == 0
        if ok:
            self._append(_("[done] Exit code 0."))
            append_recent_run(
                ModelBuilderTaskType.EVALUATE,
                getattr(self, "_pending_eval_summary", "mb evaluate"),
                True,
            )
        else:
            self._append(_("[failed] Exit code {c}.").format(c=code))
            append_recent_run(
                ModelBuilderTaskType.EVALUATE,
                getattr(self, "_pending_eval_summary", "mb evaluate"),
                False,
                f"exit {code}",
            )

    def _on_eval_error(self, message: str) -> None:
        self._append(_("[error] {m}").format(m=message))
        append_recent_run(
            ModelBuilderTaskType.EVALUATE,
            getattr(self, "_pending_eval_summary", "mb evaluate"),
            False,
            message,
        )
        qt_operation_error(
            self,
            _("Evaluate failed"),
            _("The evaluation task reported an error."),
            detail=message,
        )
