"""Small shared helpers used by ``mb.utils`` (e.g. translations)."""

from __future__ import annotations

import locale


class Utils:
    @staticmethod
    def get_default_user_language() -> str:
        try:
            loc = locale.getdefaultlocale()[0]
            if not loc:
                return "en"
            if "_" in loc:
                return loc.split("_")[0]
            return loc
        except Exception:
            return "en"
