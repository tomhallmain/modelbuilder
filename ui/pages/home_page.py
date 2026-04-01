"""Home page with summary and quick-start placeholders."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HomePage(QWidget):
    """Landing page for the desktop UI."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("<h2>Home</h2>")
        title.setTextFormat(Qt.RichText)
        layout.addWidget(title)

        subtitle = QLabel(
            "Model Builder desktop shell for data prep, training, conversion, and info tools."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        summary = QGroupBox("Current scope")
        summary_layout = QVBoxLayout(summary)
        summary_layout.addWidget(QLabel("- Data pipeline forms (element scaffolding)"))
        summary_layout.addWidget(QLabel("- Training setup forms (framework/architecture/hyperparams)"))
        summary_layout.addWidget(QLabel("- Conversion and info pages"))
        summary_layout.addWidget(QLabel("- Callback wiring and job execution in later phase"))
        layout.addWidget(summary)

        quick = QGroupBox("Quick actions")
        quick_layout = QHBoxLayout(quick)
        for text in [
            "Open Data Page",
            "Open Train Page",
            "Open Convert Page",
        ]:
            btn = QPushButton(text)
            btn.setEnabled(False)
            btn.setToolTip("Navigation callbacks not wired yet.")
            quick_layout.addWidget(btn)
        layout.addWidget(quick)

        note = QLabel("Note: Page controls are in place; command execution wiring is intentionally pending.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
