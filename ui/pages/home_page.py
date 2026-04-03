"""Home page with summary, quick navigation, and recent run history."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mb.utils.constants import ModelBuilderTaskType
from mb.utils.recent_run_history import format_recent_runs_for_display, get_recent_runs
from mb.utils.translations import _


class HomePage(QWidget):
    """Landing page for the desktop UI."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("home_page")
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(f"<h2>{_('Home')}</h2>")
        title.setTextFormat(Qt.RichText)
        layout.addWidget(title)

        subtitle = QLabel(
            _(
                "Model Builder desktop shell for data prep, training, conversion, and info tools."
            )
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        summary = QGroupBox(_("Current scope"))
        summary_layout = QVBoxLayout(summary)
        summary_layout.addWidget(
            QLabel(_("- Data pipeline: gather, convert, dedupe, upscale, create dataset"))
        )
        summary_layout.addWidget(
            QLabel(_("- Training: in-process or detached (CLI: {cmd})").format(cmd="mb train"))
        )
        summary_layout.addWidget(
            QLabel(_("- Model conversion and dataset / model inspection (Info)"))
        )
        summary_layout.addWidget(
            QLabel(_("- Application shell settings (Config)"))
        )
        layout.addWidget(summary)

        quick = QGroupBox(_("Quick actions"))
        quick_layout = QHBoxLayout(quick)
        for label, task in (
            (_("Open Data Page"), ModelBuilderTaskType.DATA),
            (_("Open Train Page"), ModelBuilderTaskType.TRAIN),
            (_("Open Convert Page"), ModelBuilderTaskType.CONVERT),
        ):
            btn = QPushButton(label)
            row = task.nav_row_index
            btn.clicked.connect(lambda checked=False, r=row: self._open_nav_row(r))
            quick_layout.addWidget(btn)
        layout.addWidget(quick)

        history = QGroupBox(_("Recent run history"))
        history_layout = QVBoxLayout(history)
        self._recent_runs = QPlainTextEdit()
        self._recent_runs.setObjectName("home_recent_runs")
        self._recent_runs.setReadOnly(True)
        self._recent_runs.setPlaceholderText(_("Loading…"))
        self._recent_runs.setMinimumHeight(180)
        history_layout.addWidget(self._recent_runs)
        layout.addWidget(history, 1)

        layout.addStretch(0)

        self.refresh_recent_runs()

    def _open_nav_row(self, row: int) -> None:
        from ui.main_window import MainWindow

        w = self.window()
        if isinstance(w, MainWindow):
            w.nav_widget.setCurrentRow(row)

    def refresh_recent_runs(self) -> None:
        text = format_recent_runs_for_display(get_recent_runs())
        self._recent_runs.setPlainText(text)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_recent_runs()
