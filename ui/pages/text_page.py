"""Text-classification page: ``mb text verify | evaluate | score | compare``.

Training lives on the Train page (model type ``text_classification``), mirroring the CLI,
where training is ``mb train --model-type text_classification``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mb.models.types import TextSubcommand
from mb.pipeline_config import get_pipeline_config
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.recent_run_history import append_recent_run
from mb.utils.translations import _
from ui.lib.fast_directory_picker_qt import get_existing_directory, get_open_file_name, get_save_file_name
from ui.lib.form_layout_i18n import apply_qform_label_column
from ui.lib.qt_alert import qt_alert, qt_operation_error
from ui.lib.task_progress import attach_progress_dialog
from ui.task_context import LongTaskContext
from ui.task_runner import start_task

_DIR = "dir"
_OPEN = "open"
_SAVE = "save"


class TextPage(QWidget):
    """Verify text datasets and evaluate, score, or compare text-classification runs."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("text_page")
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._head = QLabel()
        root.addWidget(self._head)
        self._intro = QLabel()
        self._intro.setWordWrap(True)
        root.addWidget(self._intro)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_verify_tab(), "")
        self._tabs.addTab(self._build_evaluate_tab(), "")
        self._tabs.addTab(self._build_score_tab(), "")
        self._tabs.addTab(self._build_compare_tab(), "")
        root.addWidget(self._tabs, 1)

        self.btn_verify.clicked.connect(self._run_verify)
        self.btn_evaluate.clicked.connect(self._run_evaluate)
        self.btn_score.clicked.connect(self._run_score)
        self.btn_compare.clicked.connect(self._run_compare)
        self.btn_compare_add.clicked.connect(self._add_compare_run)
        self._busy = False

        self.retranslate_ui()

    # --- construction ---

    def _output(self) -> QTextEdit:
        out = QTextEdit()
        out.setReadOnly(True)
        return out

    def _build_verify_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        group = QGroupBox(f"mb text {TextSubcommand.VERIFY.value}")
        self._verify_form = QFormLayout(group)
        self.verify_data_dir = QLineEdit()
        self.verify_data_dir.setObjectName("text_verify_data_dir_edit")
        self._verify_form.addRow(_("Data dir"), self._path_row(self.verify_data_dir, _DIR))
        v.addWidget(group)
        self.btn_verify = QPushButton()
        self.btn_verify.setObjectName("text_verify_btn")
        v.addLayout(self._button_row(self.btn_verify))
        self.verify_output = self._output()
        v.addWidget(self.verify_output, 1)
        return tab

    def _build_evaluate_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        group = QGroupBox(f"mb text {TextSubcommand.EVALUATE.value}")
        self._evaluate_form = QFormLayout(group)
        self.evaluate_run_dir = QLineEdit()
        self.evaluate_data_dir = QLineEdit()
        self._evaluate_form.addRow(_("Run dir"), self._path_row(self.evaluate_run_dir, _DIR))
        self._evaluate_form.addRow(_("Data dir (optional)"), self._path_row(self.evaluate_data_dir, _DIR))
        v.addWidget(group)
        self.btn_evaluate = QPushButton()
        v.addLayout(self._button_row(self.btn_evaluate))
        self.evaluate_output = self._output()
        v.addWidget(self.evaluate_output, 1)
        return tab

    def _build_score_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        group = QGroupBox(f"mb text {TextSubcommand.SCORE.value}")
        self._score_form = QFormLayout(group)
        self.score_run_dir = QLineEdit()
        self.score_input = QLineEdit()
        self.score_output_path = QLineEdit()
        self._score_form.addRow(_("Run dir"), self._path_row(self.score_run_dir, _DIR))
        self._score_form.addRow(_("Input lines"), self._path_row(self.score_input, _OPEN))
        self._score_form.addRow(_("Output TSV"), self._path_row(self.score_output_path, _SAVE))
        v.addWidget(group)
        self.btn_score = QPushButton()
        v.addLayout(self._button_row(self.btn_score))
        self.score_output = self._output()
        v.addWidget(self.score_output, 1)
        return tab

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        group = QGroupBox(f"mb text {TextSubcommand.COMPARE.value}")
        g = QVBoxLayout(group)
        self._compare_hint = QLabel()
        self._compare_hint.setWordWrap(True)
        g.addWidget(self._compare_hint)
        self.compare_runs = QPlainTextEdit()
        self.compare_runs.setFixedHeight(110)
        g.addWidget(self.compare_runs)
        self.btn_compare_add = QPushButton()
        g.addLayout(self._button_row(self.btn_compare_add))
        v.addWidget(group)
        self.btn_compare = QPushButton()
        v.addLayout(self._button_row(self.btn_compare))
        self.compare_output = self._output()
        self.compare_output.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.compare_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        v.addWidget(self.compare_output, 1)
        return tab

    def _button_row(self, button: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(button)
        row.addStretch(1)
        return row

    def _path_row(self, edit: QLineEdit, kind: str) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        browse = QPushButton(_("Browse..."))
        browse.clicked.connect(lambda: self._browse(edit, kind))
        h.addWidget(edit, 1)
        h.addWidget(browse, 0)
        return row

    def _browse(self, edit: QLineEdit, kind: str) -> None:
        start = edit.text().strip() or str(Path.cwd())
        if kind == _DIR:
            value = get_existing_directory(self, _("Select directory"), start)
        elif kind == _OPEN:
            value = get_open_file_name(self, _("Select file"), start, _("Text files (*.txt *.tsv);;All files (*.*)"))
        else:
            value = get_save_file_name(self, _("Save scores as"), start, _("TSV files (*.tsv);;All files (*.*)"))
        if value:
            edit.setText(value)

    def retranslate_ui(self) -> None:
        self._head.setText(f"<h2>{_('Text')}</h2>")
        self._intro.setText(
            _(
                "Text-classification datasets and runs ({cmd}). Train on the Train page with "
                "model type text_classification; defaults come from the Pipeline page's "
                "Text classification tab."
            ).format(cmd="mb text")
        )
        self._tabs.setTabText(0, _("Verify"))
        self._tabs.setTabText(1, _("Evaluate"))
        self._tabs.setTabText(2, _("Score"))
        self._tabs.setTabText(3, _("Compare"))
        apply_qform_label_column(self._verify_form, [_("Data dir")])
        apply_qform_label_column(self._evaluate_form, [_("Run dir"), _("Data dir (optional)")])
        apply_qform_label_column(self._score_form, [_("Run dir"), _("Input lines"), _("Output TSV")])
        self.verify_data_dir.setPlaceholderText(_("default: text_classification.data_dir"))
        self.evaluate_data_dir.setPlaceholderText(_("default: the run's config.yaml"))
        self.btn_verify.setText(_("Verify Dataset"))
        self.btn_evaluate.setText(_("Evaluate Run"))
        self.btn_score.setText(_("Score Lines"))
        self.btn_compare.setText(_("Compare Runs"))
        self.btn_compare_add.setText(_("Add run directory…"))
        self._compare_hint.setText(
            _("One run directory or wildcard pattern per line (e.g. data/models/text_runs/*).")
        )
        self.verify_output.setPlaceholderText(_("Integrity check results will appear here."))
        self.evaluate_output.setPlaceholderText(_("Evaluation summary will appear here."))
        self.score_output.setPlaceholderText(_("Scoring summary will appear here."))
        self.compare_output.setPlaceholderText(_("Comparison table will appear here."))
        for row_edit in (
            self.verify_data_dir,
            self.evaluate_run_dir,
            self.evaluate_data_dir,
            self.score_run_dir,
            self.score_input,
            self.score_output_path,
        ):
            row = row_edit.parentWidget()
            btn = row.findChild(QPushButton) if row is not None else None
            if btn is not None:
                btn.setText(_("Browse..."))

    def collect_gui_state(self) -> dict:
        """Serializable form state; restored by :class:`ui.controllers.cache_controller.CacheController`."""
        return {
            "tab": int(self._tabs.currentIndex()),
            "verify_data_dir": self.verify_data_dir.text(),
            "evaluate_run_dir": self.evaluate_run_dir.text(),
            "evaluate_data_dir": self.evaluate_data_dir.text(),
            "score_run_dir": self.score_run_dir.text(),
            "score_input": self.score_input.text(),
            "score_output": self.score_output_path.text(),
            "compare_runs": self.compare_runs.toPlainText(),
        }

    def restore_gui_state(self, state: dict) -> None:
        if not state:
            return
        try:
            t = state.get("tab")
            if isinstance(t, int) and 0 <= t < self._tabs.count():
                self._tabs.setCurrentIndex(t)
            self.verify_data_dir.setText(str(state.get("verify_data_dir", "")))
            self.evaluate_run_dir.setText(str(state.get("evaluate_run_dir", "")))
            self.evaluate_data_dir.setText(str(state.get("evaluate_data_dir", "")))
            self.score_run_dir.setText(str(state.get("score_run_dir", "")))
            self.score_input.setText(str(state.get("score_input", "")))
            self.score_output_path.setText(str(state.get("score_output", "")))
            self.compare_runs.setPlainText(str(state.get("compare_runs", "")))
        except Exception:
            pass

    # --- actions ---

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for b in (self.btn_verify, self.btn_evaluate, self.btn_score, self.btn_compare):
            b.setEnabled(not busy)

    def _start(
        self,
        title: str,
        summary: str,
        output: QTextEdit,
        worker: Callable[[LongTaskContext], str],
    ) -> None:
        if self._busy:
            return
        self._set_busy(True)
        output.clear()
        output.append(f"[run] {summary}")

        def on_success(text: str) -> None:
            output.append(text)
            append_recent_run(ModelBuilderTaskType.TEXT, summary, True, text.splitlines()[-1] if text else "")

        def on_error(message: str) -> None:
            output.append(_("[error] {err}").format(err=message))
            append_recent_run(ModelBuilderTaskType.TEXT, summary, False, message)
            qt_operation_error(
                self,
                title,
                _("The operation stopped with an error. See Details for the message."),
                detail=message,
            )

        def on_cancelled() -> None:
            output.append(_("[stopped] Cancelled."))
            append_recent_run(ModelBuilderTaskType.TEXT, summary, False, "cancelled")

        handle = start_task(
            worker,
            on_success,
            on_error,
            lambda: self._set_busy(False),
            pass_context=True,
            on_cancelled=on_cancelled,
        )
        attach_progress_dialog(self, title, handle, cancellable=True)

    def _require(self, edit: QLineEdit, what: str) -> Path | None:
        text = edit.text().strip()
        if not text:
            qt_alert(self, _("Missing input"), _("Please provide: {what}").format(what=what), kind="warning")
            return None
        return Path(text)

    def _run_verify(self) -> None:
        from mb.data.text_dataset import verify_text_dataset
        from mb.training.text_config import dataset_file_options

        raw = self.verify_data_dir.text().strip() or str(get_pipeline_config().get("text_classification.data_dir") or "")
        data_dir = Path(raw)
        if not data_dir.is_dir():
            qt_alert(
                self,
                _("Missing directory"),
                _("Data directory not found: {path}").format(path=data_dir),
                kind="warning",
            )
            return

        file_options = dataset_file_options(get_pipeline_config())

        def work(ctx: LongTaskContext) -> str:
            report = verify_text_dataset(
                data_dir,
                cancel_event=ctx.cancel_event,
                progress=lambda m: ctx.progress(m, None),
                **file_options,
            )
            if not report.ok:
                raise ValueError(report.format())
            return report.format()

        self._start(_("Verify dataset"), f"mb text verify --data-dir {data_dir}", self.verify_output, work)

    def _run_evaluate(self) -> None:
        from mb.evaluate.classification.text_evaluation import METRICS_FILE, evaluate_text_run

        run_dir = self._require(self.evaluate_run_dir, _("Run dir"))
        if run_dir is None:
            return
        data_raw = self.evaluate_data_dir.text().strip()
        data_dir = Path(data_raw) if data_raw else None

        def work(ctx: LongTaskContext) -> str:
            result = evaluate_text_run(
                run_dir, data_dir=data_dir, cancel_event=ctx.cancel_event, progress=lambda m, p: ctx.progress(m, p)
            )
            gold = result.metrics["test"]["gold"]
            full = result.metrics["test"]["full"]

            def f(x):
                return "-" if x is None else f"{x:.4f}"

            return "\n".join(
                [
                    _("Test gold AP: {g} — full AP: {f} — decision threshold: {t:.4f}").format(
                        g=f(gold.get("ap")), f=f(full.get("ap")), t=result.thresholds.decision
                    ),
                    _("Wrote {path}").format(path=result.run_dir / METRICS_FILE),
                ]
            )

        self._start(_("Evaluate run"), f"mb text evaluate --model {run_dir}", self.evaluate_output, work)

    def _run_score(self) -> None:
        from mb.evaluate.classification.text_evaluation import score_text_file

        run_dir = self._require(self.score_run_dir, _("Run dir"))
        input_path = self._require(self.score_input, _("Input lines")) if run_dir else None
        output_path = self._require(self.score_output_path, _("Output TSV")) if input_path else None
        if output_path is None:
            return

        def work(ctx: LongTaskContext) -> str:
            result = score_text_file(
                run_dir, input_path, output_path, cancel_event=ctx.cancel_event, progress=lambda m, p: ctx.progress(m, p)
            )
            lines = [_("Scored {n} lines -> {path}").format(n=result.n_lines, path=result.output)]
            if result.n_with_tab:
                lines.insert(
                    0,
                    _("[warn] {n} input line(s) contain a tab; their output rows have extra columns.").format(
                        n=result.n_with_tab
                    ),
                )
            return "\n".join(lines)

        self._start(
            _("Score lines"),
            f"mb text score --model {run_dir} --input {input_path} --output {output_path}",
            self.score_output,
            work,
        )

    def _add_compare_run(self) -> None:
        value = get_existing_directory(self, _("Select run directory"), str(Path.cwd()))
        if value:
            self.compare_runs.appendPlainText(value)

    def _run_compare(self) -> None:
        import glob

        from mb.evaluate.classification.text_evaluation import compare_text_runs, format_compare_table

        dirs: list[Path] = []
        for line in self.compare_runs.toPlainText().splitlines():
            pattern = line.strip()
            if not pattern:
                continue
            matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
            dirs.extend(Path(m) for m in matches if Path(m).is_dir())
        rows, skipped = compare_text_runs(dirs)
        self.compare_output.clear()
        for d in skipped:
            self.compare_output.append(_("[skip] no metrics.json: {path}").format(path=d))
        if not rows:
            self.compare_output.append(_("No run directories with metrics.json were found."))
            return
        self.compare_output.append(format_compare_table(rows))
