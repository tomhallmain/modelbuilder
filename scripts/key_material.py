"""
Inspect or back up this application's encryption key material.

``status`` (the default) is read-only and prints no secrets -- it reports
presence, location and freshness so you can answer "where is it, has the
migration finished, and is my off-drive backup current?" without opening
anything that contains a key.

``backup`` writes a full copy, passphrase included, for the cases automatic
backup cannot cover: seeding a new machine, or capturing state before a
change. Routine backups happen on their own whenever the key store is
written (AUTO_BACKUP_KEY_STORE in utils/encryptor.py).

Usage:
  python scripts/key_material.py                       # status
  python scripts/key_material.py status --include-legacy
  python scripts/key_material.py backup                # to the detected drive
  python scripts/key_material.py backup -o E:/keys.json
  python scripts/key_material.py backup --stdout       # print, write nothing
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from mb.utils.constants import AppInfo  # noqa: E402
from mb.utils.translations import _  # noqa: E402
from utils.encryptor import (  # noqa: E402
    AUTO_BACKUP_KEY_STORE,
    ENCRYPTOR_TYPE_KEY,
    KEY_BACKUP_DIR_ENV_VAR,
    KEY_STORE_DIR_ENV_VAR,
    PasswordManager,
    available_external_drives,
    default_key_backup_path,
    export_key_material,
    get_key_base,
    has_prior_key_material,
    key_store_path,
    namespaced_key,
    peek_passphrase,
    read_key_store,
)
import keyring  # noqa: E402

_LABEL_WIDTH = 12
_NOTE_INDENT = " " * (2 + _LABEL_WIDTH + 2)


def _row(label: str, value: str) -> None:
    """One ``  label : value`` line of the status report."""
    print("  {label:<{width}}: {value}".format(label=label, width=_LABEL_WIDTH, value=value))


def _note(text: str) -> None:
    """A continuation note under the previous row, wrapped to the report's indent."""
    print(textwrap.fill(
        text, width=72, initial_indent=_NOTE_INDENT, subsequent_indent=_NOTE_INDENT
    ))


def _identifiers(include_legacy: bool) -> list:
    """Current app identifier, plus the legacy ones when asked.

    A renamed app leaves key material filed under its old identifier, and that
    material stays load-bearing until everything encrypted under it has been
    migrated. A rename only has to add the old name to
    AppInfo.LEGACY_APP_IDENTIFIERS for ``--include-legacy`` to cover it.
    """
    identifiers = [AppInfo.APP_IDENTIFIER]
    if include_legacy:
        identifiers.extend(
            app_id for app_id in AppInfo.LEGACY_APP_IDENTIFIERS
            if app_id not in identifiers
        )
    return identifiers


def _mtime(path: str) -> str:
    try:
        stamp = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        return stamp.strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return _("unknown")


def _legacy_items_present(service_name: str, app_identifier: str) -> list:
    """Pre-consolidation keychain items still in place.

    Anything here means migration has not finished for this identifier -- the
    keys are still spread across one keychain item per component.
    """
    found = []
    for key in ("salt", "nonce", "tag", ENCRYPTOR_TYPE_KEY):
        if keyring.get_password(service_name, namespaced_key(app_identifier, key)):
            found.append(key)
    # Stored passwords would also be chunked items, migrated lazily on first
    # retrieve, but this app stores none.
    for base in ("encrypted_priv", "public_key"):
        count_key = namespaced_key(get_key_base(app_identifier, base), "count")
        count = keyring.get_password(service_name, count_key)
        if count:
            found.append(_("{name} ({count} chunks)").format(name=base, count=count))
    return found


def _report_identifier(service_name: str, app_identifier: str) -> None:
    print(f"\n--- {app_identifier} ---")
    if not has_prior_key_material(service_name, app_identifier):
        print("  " + _("No key material found for this identifier."))
        return

    store_path = key_store_path(service_name, app_identifier)
    store = read_key_store(service_name, app_identifier)
    _row(_("key store"), store_path)
    if store is None:
        _note(_(
            "MISSING or unreadable -- the app will refuse to generate new keys "
            "and ask you to restore a backup."
        ))
    else:
        passwords = sorted(store.get(PasswordManager.STORE_SECTION, {}))
        _note(_("present, {size} bytes, modified {mtime}").format(
            size=os.path.getsize(store_path), mtime=_mtime(store_path)
        ))
        _row(_("encryptor"), store.get(ENCRYPTOR_TYPE_KEY) or _("unknown"))
        _row(_("passwords"), ", ".join(passwords) if passwords else _("(none stored)"))

    _row(
        _("passphrase"),
        _("present in keychain") if peek_passphrase(service_name, app_identifier)
        else _("ABSENT"),
    )

    legacy = _legacy_items_present(service_name, app_identifier)
    if legacy:
        _row(_("legacy items"), ", ".join(legacy))
        _note(_(
            "migration has not completed for this identifier; starting the app "
            "should finish it."
        ))
    else:
        _row(_("legacy items"), _("none (migration complete)"))

    backup = default_key_backup_path(service_name, app_identifier)
    if not backup:
        _row(_("backup"), _("no destination -- nothing is being backed up"))
    elif not os.path.exists(backup):
        _row(_("backup"), backup)
        _note(_(
            "NOT YET WRITTEN (auto-backup runs on the next key store write; "
            "'backup' below writes one now)"
        ))
    else:
        _row(_("backup"), backup)
        _note(_("written {mtime}").format(mtime=_mtime(backup)))
        try:
            with open(backup, "r", encoding="utf-8") as handle:
                backed_up = json.load(handle)
            if backed_up.get("key_store") == store:
                _note(_("up to date with the key store"))
            elif backed_up.get("key_store") is None:
                _note(_(
                    "PREDATES MIGRATION (holds legacy items, not a key store) -- "
                    "still restorable, but stale"
                ))
            else:
                _note(_("STALE -- differs from the current key store"))
        except Exception as e:
            _note(_("could not be read back: {error}").format(error=e))


def cmd_status(args: argparse.Namespace) -> int:
    print(_("Service: {name}").format(name=AppInfo.SERVICE_NAME))
    print(_("Auto-backup: {state}").format(
        state=_("on") if AUTO_BACKUP_KEY_STORE else _("off")
    ))
    for name in (KEY_BACKUP_DIR_ENV_VAR, KEY_STORE_DIR_ENV_VAR):
        print(f"{name}: {os.environ.get(name) or _('(unset)')}")
    drives = available_external_drives()
    print(_("External drives: {drives}").format(
        drives=", ".join(drives) if drives else _("(none detected)")
    ))

    for app_identifier in _identifiers(args.include_legacy):
        _report_identifier(AppInfo.SERVICE_NAME, app_identifier)
    return 0


def _resolve_output(explicit: str | None, app_identifier: str) -> str | None:
    if explicit:
        # One file per identifier when several are exported to one path.
        stem, ext = os.path.splitext(explicit)
        return explicit if app_identifier == AppInfo.APP_IDENTIFIER else (
            f"{stem}__{app_identifier}{ext or '.json'}"
        )
    return default_key_backup_path(AppInfo.SERVICE_NAME, app_identifier)


def cmd_backup(args: argparse.Namespace) -> int:
    print(_("Service: {name}").format(name=AppInfo.SERVICE_NAME))
    written = []
    for app_identifier in _identifiers(args.include_legacy):
        print(f"\n--- {app_identifier} ---")
        if not has_prior_key_material(AppInfo.SERVICE_NAME, app_identifier):
            print(_("No key material found; nothing to back up."))
            continue

        if args.stdout:
            export_key_material(AppInfo.SERVICE_NAME, app_identifier)
            continue

        output = _resolve_output(args.output, app_identifier)
        if not output:
            print(_(
                "No external drive detected and no -o given. Re-run with "
                "-o <path on another drive>, or --stdout to print instead."
            ))
            return 1

        directory = os.path.dirname(output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        material = export_key_material(AppInfo.SERVICE_NAME, app_identifier, output)
        # Read back rather than trusting the write: an unreadable backup is
        # indistinguishable from no backup at the moment it is needed.
        with open(output, "r", encoding="utf-8") as handle:
            if json.load(handle) != material:
                print(_(
                    "WARNING: {path} did not read back intact -- do not rely on it."
                ).format(path=output))
                return 1
        written.append(output)

    if written:
        print("\n" + _("Verified backups written:"))
        for path in written:
            print(f"  {path}")
        print("\n" + _("Each contains the keychain passphrase in the clear. Restore with:"))
        print("  from utils.encryptor import import_key_material")
        print("  import_key_material(json.load(open(<path>)))")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    legacy_help = _("Also cover legacy identifiers: {identifiers}").format(
        identifiers=", ".join(AppInfo.LEGACY_APP_IDENTIFIERS) or _("(none)")
    )

    status = sub.add_parser("status", help=_("Report state; prints no secrets."))
    status.add_argument("--include-legacy", action="store_true", help=legacy_help)
    status.set_defaults(func=cmd_status)

    backup = sub.add_parser("backup", help=_("Write a full copy, passphrase included."))
    backup.add_argument(
        "-o", "--output", help=_("File to write. Omit for the detected drive.")
    )
    backup.add_argument("--stdout", action="store_true", help=_("Print instead of writing."))
    backup.add_argument("--include-legacy", action="store_true", help=legacy_help)
    backup.set_defaults(func=cmd_backup)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        # Bare invocation reports state -- the harmless default.
        args = parser.parse_args(["status"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
