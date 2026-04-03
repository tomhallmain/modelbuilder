"""
Human-readable remaining time strings for training progress UI.
"""

from __future__ import annotations


def format_eta_seconds(seconds: float) -> str:
    """
    Format a remaining duration for display (English; wrap with gettext in UI if needed).

    Uses days/hours when appropriate; sub-minute uses seconds; avoids "0 hours".
    """
    if seconds != seconds or seconds < 0:  # NaN or negative
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s" if s != 1 else "1s"
    m, s = divmod(s, 60)
    if m < 60:
        if s == 0:
            return f"{m} min" if m != 1 else "1 min"
        return f"{m} min {s}s"
    h, m = divmod(m, 60)
    if h < 24:
        if m == 0:
            return f"{h} h" if h != 1 else "1 h"
        return f"{h} h {m} min"
    d, h = divmod(h, 24)
    if h == 0:
        return f"{d} days" if d != 1 else "1 day"
    return f"{d} days {h} h"


def format_eta_phrase(seconds: float) -> str:
    """e.g. 'About 2 h 15 min remaining' or em dash when unknown."""
    inner = format_eta_seconds(seconds)
    if inner == "—":
        return inner
    return f"About {inner} remaining"
