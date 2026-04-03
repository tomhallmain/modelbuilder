import gettext
import os
from pathlib import Path

from mb.utils.logging_setup import get_logger
from mb.utils.utils import Utils

logger = get_logger("translations")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Locales offered in the desktop Config page. Other YAML values are treated as unsupported
# (combo falls back to “system default” until the user picks en/de).
SUPPORTED_GUI_LOCALES: tuple[str, ...] = ("en", "de")


def normalize_gui_locale(locale: object) -> str | None:
    """
    Map a config or environment locale string to ``\"en\"``, ``\"de\"``, or ``None``.

    ``None`` means use the OS / application default (empty ``gui.locale``).
    Unknown region codes (e.g. ``de_AT``) map to the primary language when it is supported.
    """
    if locale is None or str(locale).strip() == "":
        return None
    s = str(locale).strip().lower().replace("-", "_")
    short = s.split("_", 1)[0] if "_" in s else s
    if short in SUPPORTED_GUI_LOCALES:
        return short
    return None


def _short_locale(locale: str) -> str:
    loc = locale.replace("-", "_").strip()
    if "_" in loc:
        return loc.split("_", 1)[0].lower()
    return loc.lower() if loc else "en"


def apply_application_locale(*, verbose: bool = False) -> None:
    """
    Install gettext locale from application config (``gui.locale``).

    Updates ``LANG`` so subprocesses and code using the environment see the same value.
    """
    from utils.config import get_application_config

    loc = get_application_config().gui.locale
    if loc is None or str(loc).strip() == "":
        loc_full = str(Utils.get_default_user_language()).strip()
    else:
        norm = normalize_gui_locale(loc)
        if norm is not None:
            loc_full = norm
        else:
            # Legacy or unsupported YAML value: follow OS default
            loc_full = str(Utils.get_default_user_language()).strip()
    os.environ["LANG"] = loc_full
    short = _short_locale(loc_full)
    I18N.install_locale(short, verbose=verbose)


_locale = os.environ.get("LANG") or os.environ.get("LANGUAGE")
if not _locale or _locale == "":
    _locale = Utils.get_default_user_language()
elif "_" in _locale:
    _locale = _locale.split("_")[0]

class I18N:
    localedir = str(_REPO_ROOT / "locale")
    locale = "en"
    translate = gettext.translation(
        "base", localedir, languages=[_locale or "en"], fallback=True
    )

    @staticmethod
    def install_locale(locale, verbose=True):
        I18N.locale = locale
        I18N.translate = gettext.translation('base', I18N.localedir, languages=[locale], fallback=True)
        I18N.translate.install()
        if verbose:
            logger.info("Switched locale to: " + locale)

    @staticmethod
    def _(s):
        # return gettext.gettext(s)
        try:
            return I18N.translate.gettext(s)
        except KeyError:
            return s

    '''
    NOTE when gathering the translation strings, set _() == to gettext.gettext() instead of the above, and run:

        ```python C:\\Python310\\Tools\\i18n\\pygettext.py -d base -o locale\\base.pot .```

    in the base directory. The POT output file can be used as source for the PO files in each locale.
    Run personal script C:\\Scripts\\i18n_manager.py to generate new PO files and look for invalid translations.

    Bonus command:
        ```git diff locale\\de\\LC_MESSAGES\\base.po locale\\de\\LC_MESSAGES\\base1.po | rg -v "^.*#" | rg -C 3 "^(-|\\+)"```

    Then for each locale once the PO files are set up as desired, run below in the deepest locale directory to produce the MO file from the PO file:
        ```python C:\\Python310\\Tools\\i18n\\msgfmt.py -o base.mo base```
    '''


def _(message: str) -> str:
    """
    Translate *message* for the active locale.

    Same as :meth:`I18N._`; prefer ``from mb.utils.translations import _`` in callers
    instead of ``_ = I18N._``.
    """
    return I18N._(message)
