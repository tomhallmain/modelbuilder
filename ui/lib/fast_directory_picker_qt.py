"""
Fast cross-platform path pickers for Qt (directories and files).

This avoids Windows native QFileDialog latency by using lightweight custom
dialogs that only scan the **currently viewed** folder (and drive roots via
bitmask APIs on Windows). Open/save file pickers list files in the current
directory only—no recursive or “whole volume” enumeration.

Safety and persistence notes:
- This module performs no destructive filesystem operations (except explicit
  “New folder” in directory mode, and save path selection does not write files).
- Cache state is in-memory only (process lifetime) and never written to disk.
"""

from __future__ import annotations

import os
import platform
import time
from threading import RLock

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import logging

from mb.utils.translations import _

logger = logging.getLogger(__name__)


def parse_qt_name_filter(name_filter: str) -> list[str] | None:
    """
    Parse a Qt-style filter string (e.g. ``YAML (*.yaml *.yml);;All files (*.*)``).

    Returns:
        ``None`` to show all files in the current directory, or a sorted list of
        lowercase extensions including the leading dot (e.g. ``['.yaml', '.yml']``).
    """
    if not name_filter or not str(name_filter).strip():
        return None
    has_glob_all = False
    exts: set[str] = set()
    for segment in str(name_filter).split(";;"):
        segment = segment.strip()
        if "(" not in segment or ")" not in segment:
            continue
        inner = segment[segment.index("(") + 1 : segment.rindex(")")]
        for token in inner.split():
            t = token.strip().lower()
            if t in ("*", "*.*"):
                has_glob_all = True
                continue
            if t.startswith("*."):
                exts.add(t[1:])
            elif t.startswith("."):
                exts.add(t)
    if has_glob_all and not exts:
        return None
    if has_glob_all and exts:
        return None
    return sorted(exts) if exts else None


def list_matching_files(
    directory: str, extensions: list[str] | None
) -> list[tuple[str, str]]:
    """
    List regular files in *directory* only (no recursion).

    Returns:
        Sorted ``(absolute_path, basename)`` pairs.
    """
    out: list[tuple[str, str]] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                name = entry.name
                if extensions is None:
                    out.append((os.path.normpath(entry.path), name))
                    continue
                low = name.lower()
                if any(low.endswith(ext) for ext in extensions):
                    out.append((os.path.normpath(entry.path), name))
    except OSError as e:
        logger.debug("Cannot list files in '%s': %s", directory, e)
    out.sort(key=lambda x: x[1].casefold())
    return out


def _center_dialog_on_parent_or_screen(dialog: QDialog, parent: QWidget | None) -> None:
    dialog.adjustSize()
    if parent is not None:
        pw = parent.window()
        if pw is not None:
            geo = dialog.frameGeometry()
            geo.moveCenter(pw.frameGeometry().center())
            dialog.move(geo.topLeft())
            return
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return
    avail = screen.availableGeometry()
    geo = dialog.frameGeometry()
    geo.moveCenter(avail.center())
    dialog.move(geo.topLeft())

# Session-only context memory for better default starting location.
_session_last_directory_hint: str = ""
_session_has_opened_picker: bool = False


class _DirectoryPickerCache:
    """Small in-memory cache for roots and subdirectory listings."""

    _lock = RLock()
    _roots_cache: tuple[float, list[tuple[str, str]]] | None = None
    _subdirs_cache: dict[str, tuple[float, list[str]]] = {}

    ROOTS_TTL_SECONDS = 5 * 60
    SUBDIRS_TTL_SECONDS = 30
    SUBDIRS_MAX_ENTRIES = 256

    @classmethod
    def get_roots(cls) -> list[tuple[str, str]]:
        now = time.time()
        with cls._lock:
            if cls._roots_cache and (now - cls._roots_cache[0]) < cls.ROOTS_TTL_SECONDS:
                return list(cls._roots_cache[1])

        roots = cls._compute_roots()
        with cls._lock:
            cls._roots_cache = (now, roots)
        return list(roots)

    @classmethod
    def invalidate_roots(cls) -> None:
        with cls._lock:
            cls._roots_cache = None

    @classmethod
    def get_subdirs(cls, directory: str) -> list[str]:
        normalized = os.path.normpath(directory)
        now = time.time()
        with cls._lock:
            cached = cls._subdirs_cache.get(normalized)
            if cached and (now - cached[0]) < cls.SUBDIRS_TTL_SECONDS:
                return list(cached[1])

        children: list[str] = []
        try:
            with os.scandir(normalized) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            children.append(entry.path)
                    except OSError:
                        # Skip inaccessible entries; don't fail the whole listing.
                        continue
        except OSError as e:
            logger.debug(f"Cannot scan directory '{normalized}': {e}")

        children.sort(key=lambda p: os.path.basename(os.path.normpath(p)).casefold())

        with cls._lock:
            cls._subdirs_cache[normalized] = (now, children)
            if len(cls._subdirs_cache) > cls.SUBDIRS_MAX_ENTRIES:
                # Drop oldest cache entries first.
                oldest_keys = sorted(
                    cls._subdirs_cache.keys(),
                    key=lambda k: cls._subdirs_cache[k][0],
                )[: len(cls._subdirs_cache) - cls.SUBDIRS_MAX_ENTRIES]
                for key in oldest_keys:
                    cls._subdirs_cache.pop(key, None)
        return children

    @classmethod
    def invalidate_subdirs(cls, directory: str | None = None) -> None:
        with cls._lock:
            if directory is None:
                cls._subdirs_cache.clear()
                return
            cls._subdirs_cache.pop(os.path.normpath(directory), None)

    @staticmethod
    def _compute_roots() -> list[tuple[str, str]]:
        system = platform.system().lower()
        if system == "windows":
            return _DirectoryPickerCache._compute_windows_roots()
        return _DirectoryPickerCache._compute_posix_roots()

    @staticmethod
    def _compute_windows_roots() -> list[tuple[str, str]]:
        roots: list[tuple[str, str]] = []
        seen_roots: set[str] = set()

        home = os.path.expanduser("~")
        if home and os.path.isdir(home):
            normalized_home = os.path.normpath(home)
            roots.append((normalized_home, _("Home")))
            seen_roots.add(normalized_home.casefold())

        try:
            import ctypes

            drives_mask = ctypes.windll.kernel32.GetLogicalDrives()
            get_drive_type = ctypes.windll.kernel32.GetDriveTypeW

            drive_type_names = {
                2: _("Removable"),
                3: _("Local"),
                4: _("Network"),
                5: _("CD/DVD"),
                6: _("RAM Disk"),
            }

            for index in range(26):
                if not (drives_mask & (1 << index)):
                    continue
                letter = chr(ord("A") + index)
                root = f"{letter}:\\"
                normalized_root = os.path.normpath(root)
                if normalized_root.casefold() in seen_roots:
                    continue
                drive_type = int(get_drive_type(root))
                drive_type_text = drive_type_names.get(drive_type, _("Unknown"))
                label = f"{root} ({drive_type_text})"
                roots.append((normalized_root, label))
                seen_roots.add(normalized_root.casefold())
        except Exception as e:
            logger.error(f"Failed to enumerate Windows drives: {e}")

        if not roots:
            roots = [("C:\\", "C:\\")]
        return roots

    @staticmethod
    def _compute_posix_roots() -> list[tuple[str, str]]:
        """
        Enumerate useful mount points for Linux/macOS/BSD.

        Uses psutil when available (fast and cross-platform), then falls back
        to common roots so picker behavior remains robust without extras.
        """
        roots: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add_root(path: str, label: str | None = None) -> None:
            if not path:
                return
            norm = os.path.normpath(path)
            if not norm or norm in seen or not os.path.isdir(norm):
                return
            seen.add(norm)
            roots.append((norm, label or norm))

        # Always include canonical roots first for predictable UX.
        add_root("/", "/")
        home = os.path.expanduser("~")
        if home and home != "/":
            add_root(home, _("Home"))

        # Include mounted volumes from psutil when available.
        try:
            import psutil

            for part in psutil.disk_partitions(all=False):
                mountpoint = (part.mountpoint or "").strip()
                if not mountpoint:
                    continue
                fs = (part.fstype or "").lower()
                opts = (part.opts or "").lower()
                device = (part.device or "").strip()

                # Skip pseudo/system mounts that add noise and are rarely useful.
                if fs in {"proc", "sysfs", "devtmpfs", "devfs", "tmpfs", "overlay", "squashfs"}:
                    continue
                if "loop" in device and "rw" not in opts:
                    continue

                is_network = fs in {"nfs", "cifs", "smbfs", "sshfs"} or "://" in device
                if is_network:
                    label = f"{mountpoint} ({_('Network')})"
                else:
                    label = mountpoint
                add_root(mountpoint, label)
        except Exception as e:
            logger.debug(f"psutil mount enumeration unavailable: {e}")

        # Common external-media parent directories as fallbacks.
        for base in ("/Volumes", "/media", "/mnt", "/run/media"):
            if not os.path.isdir(base):
                continue
            add_root(base, base)
            try:
                with os.scandir(base) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            add_root(entry.path, entry.path)
            except OSError:
                continue

        return roots if roots else [("/", "/")]


class FastDirectoryPickerDialog(QDialog):
    """Custom, efficient directory picker that avoids native file dialog IO."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        title: str,
        initial_dir: str = "",
        quick_access_locations: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or _("Select Directory"))
        self.resize(860, 560)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.selected_directory = ""
        self._current_directory = ""
        self._quick_access_locations = (
            list(quick_access_locations) if quick_access_locations else []
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        content = QHBoxLayout()
        content.setSpacing(8)
        outer.addLayout(content, 1)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        content.addLayout(left_col, 1)

        left_col.addWidget(QLabel(_("Locations")))
        self._roots_list = QListWidget()
        self._roots_list.itemDoubleClicked.connect(self._on_root_double_clicked)
        left_col.addWidget(self._roots_list, 1)

        refresh_roots_btn = QPushButton(_("Refresh locations"))
        refresh_roots_btn.clicked.connect(self._refresh_roots)
        left_col.addWidget(refresh_roots_btn)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        content.addLayout(right_col, 3)

        path_bar = QHBoxLayout()
        path_bar.setSpacing(6)
        right_col.addLayout(path_bar)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(_("Type or paste a directory path"))
        self._path_edit.returnPressed.connect(self._go_to_path)
        path_bar.addWidget(self._path_edit, 1)

        go_btn = QPushButton(_("Go"))
        go_btn.clicked.connect(self._go_to_path)
        path_bar.addWidget(go_btn)

        up_btn = QPushButton(_("Up"))
        up_btn.clicked.connect(self._go_up)
        path_bar.addWidget(up_btn)

        new_dir_btn = QPushButton(_("New Folder"))
        new_dir_btn.clicked.connect(self._create_directory)
        path_bar.addWidget(new_dir_btn)

        refresh_btn = QPushButton(_("Refresh"))
        refresh_btn.clicked.connect(self._refresh_current_directory)
        path_bar.addWidget(refresh_btn)

        right_col.addWidget(QLabel(_("Folders")))
        self._subdirs_list = QListWidget()
        self._subdirs_list.itemDoubleClicked.connect(self._on_subdir_double_clicked)
        right_col.addWidget(self._subdirs_list, 1)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #ff7777;")
        self._status_label.setVisible(False)
        right_col.addWidget(self._status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        outer.addLayout(buttons)

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        select_btn = QPushButton(_("Select"))
        select_btn.clicked.connect(self._select_directory)
        select_btn.setDefault(True)
        buttons.addWidget(select_btn)

        self._load_roots()
        self._set_initial_directory(initial_dir)
        _center_dialog_on_parent_or_screen(self, parent)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Consume Escape in this dialog so parent Escape shortcuts do not fire.
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            self.reject()
            return
        super().keyPressEvent(event)

    def _load_roots(self) -> None:
        self._roots_list.clear()
        seen: set[str] = set()
        all_locations = list(self._quick_access_locations) + _DirectoryPickerCache.get_roots()
        for root, label in all_locations:
            normalized = os.path.normpath(str(root))
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, normalized)
            self._roots_list.addItem(item)

    def _set_initial_directory(self, initial_dir: str) -> None:
        target = initial_dir.strip() if initial_dir else ""
        if target:
            # NOTE: We start at the previous selection's parent since
            # it is common for specifically-selected directories to be
            # leaves in the file hierarchy, and we want to present the
            #  user with more options to branch off of on first load.
            normalized = os.path.normpath(target)
            if not os.path.isdir(normalized):
                normalized = os.path.dirname(normalized)
            parent = os.path.dirname(normalized)
            target = parent if parent else normalized
        if not target:
            roots = _DirectoryPickerCache.get_roots()
            target = roots[0][0] if roots else os.path.expanduser("~")
        self._navigate_to_directory(target)

    def _refresh_roots(self) -> None:
        _DirectoryPickerCache.invalidate_roots()
        self._load_roots()

    def _refresh_current_directory(self) -> None:
        if not self._current_directory:
            return
        _DirectoryPickerCache.invalidate_subdirs(self._current_directory)
        self._populate_subdirs()

    def _create_directory(self) -> None:
        if not self._current_directory:
            return
        folder_name, ok = QInputDialog.getText(
            self,
            _("Create New Folder"),
            _("Folder name:"),
        )
        if not ok:
            return
        folder_name = folder_name.strip()
        if not folder_name:
            self._status_label.setText(_("Folder name cannot be empty."))
            self._status_label.setVisible(True)
            return
        target_path = os.path.normpath(
            os.path.join(self._current_directory, folder_name)
        )
        try:
            os.makedirs(target_path, exist_ok=False)
            _DirectoryPickerCache.invalidate_subdirs(self._current_directory)
            _DirectoryPickerCache.invalidate_subdirs(target_path)
            self._navigate_to_directory(target_path)
        except FileExistsError:
            self._status_label.setText(_("A folder with that name already exists."))
            self._status_label.setVisible(True)
        except OSError as e:
            logger.warning(f"Failed to create directory '{target_path}': {e}")
            self._status_label.setText(
                _("Failed to create folder (check permissions/path).")
            )
            self._status_label.setVisible(True)

    def _on_root_double_clicked(self, item: QListWidgetItem) -> None:
        root = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if root:
            self._navigate_to_directory(root)

    def _on_subdir_double_clicked(self, item: QListWidgetItem) -> None:
        directory = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if directory:
            self._navigate_to_directory(directory)

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip()
        if path:
            self._navigate_to_directory(path)

    def _go_up(self) -> None:
        if not self._current_directory:
            return
        parent = os.path.dirname(os.path.normpath(self._current_directory))
        if not parent:
            return
        self._navigate_to_directory(parent)

    def _navigate_to_directory(self, directory: str) -> None:
        normalized = os.path.normpath(directory.strip()) if directory else ""
        if not normalized:
            return

        self._current_directory = normalized
        self._path_edit.setText(normalized)
        self._populate_subdirs()

    def _populate_subdirs(self) -> None:
        self._subdirs_list.clear()
        self._status_label.setVisible(False)
        children = _DirectoryPickerCache.get_subdirs(self._current_directory)
        for child in children:
            name = os.path.basename(os.path.normpath(child)) or child
            item = QListWidgetItem(name)
            item.setToolTip(child)
            item.setData(Qt.ItemDataRole.UserRole, child)
            self._subdirs_list.addItem(item)

        if not children:
            self._status_label.setText(
                _("No folders found (or folder is not currently accessible).")
            )
            self._status_label.setVisible(True)

    def _select_directory(self) -> None:
        current_item = self._subdirs_list.currentItem()
        if current_item is not None:
            candidate = str(current_item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if candidate:
                self.selected_directory = os.path.normpath(candidate)
                self.accept()
                return

        typed = self._path_edit.text().strip()
        if typed:
            self.selected_directory = os.path.normpath(typed)
            self.accept()
            return

        if self._current_directory:
            self.selected_directory = os.path.normpath(self._current_directory)
            self.accept()


class FastOpenFilePickerDialog(QDialog):
    """Efficient open-file picker: only lists the current directory (no native dialog)."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        title: str,
        initial_path: str = "",
        name_filter: str = "",
        quick_access_locations: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or _("Open File"))
        self.resize(900, 620)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.selected_file = ""
        self._current_directory = ""
        self._pending_select_basename: str | None = None
        self._exts = parse_qt_name_filter(name_filter)
        self._quick_access_locations = (
            list(quick_access_locations) if quick_access_locations else []
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        content = QHBoxLayout()
        content.setSpacing(8)
        outer.addLayout(content, 1)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        content.addLayout(left_col, 1)

        left_col.addWidget(QLabel(_("Locations")))
        self._roots_list = QListWidget()
        self._roots_list.itemDoubleClicked.connect(self._on_root_double_clicked)
        left_col.addWidget(self._roots_list, 1)

        refresh_roots_btn = QPushButton(_("Refresh locations"))
        refresh_roots_btn.clicked.connect(self._refresh_roots)
        left_col.addWidget(refresh_roots_btn)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        content.addLayout(right_col, 3)

        path_bar = QHBoxLayout()
        path_bar.setSpacing(6)
        right_col.addLayout(path_bar)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(_("Type or paste a directory or file path"))
        self._path_edit.returnPressed.connect(self._go_to_path)
        path_bar.addWidget(self._path_edit, 1)

        go_btn = QPushButton(_("Go"))
        go_btn.clicked.connect(self._go_to_path)
        path_bar.addWidget(go_btn)

        up_btn = QPushButton(_("Up"))
        up_btn.clicked.connect(self._go_up)
        path_bar.addWidget(up_btn)

        new_dir_btn = QPushButton(_("New Folder"))
        new_dir_btn.clicked.connect(self._create_directory)
        path_bar.addWidget(new_dir_btn)

        refresh_btn = QPushButton(_("Refresh"))
        refresh_btn.clicked.connect(self._refresh_current_directory)
        path_bar.addWidget(refresh_btn)

        right_col.addWidget(QLabel(_("Folders")))
        self._subdirs_list = QListWidget()
        self._subdirs_list.itemDoubleClicked.connect(self._on_subdir_double_clicked)
        right_col.addWidget(self._subdirs_list, 1)

        right_col.addWidget(QLabel(_("Files")))
        self._files_list = QListWidget()
        self._files_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        right_col.addWidget(self._files_list, 1)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #ff7777;")
        self._status_label.setVisible(False)
        right_col.addWidget(self._status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        outer.addLayout(buttons)

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        open_btn = QPushButton(_("Open"))
        open_btn.clicked.connect(self._accept_file)
        open_btn.setDefault(True)
        buttons.addWidget(open_btn)

        self._load_roots()
        self._set_initial_path(initial_path)
        _center_dialog_on_parent_or_screen(self, parent)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            self.reject()
            return
        super().keyPressEvent(event)

    def _load_roots(self) -> None:
        self._roots_list.clear()
        seen: set[str] = set()
        all_locations = list(self._quick_access_locations) + _DirectoryPickerCache.get_roots()
        for root, label in all_locations:
            normalized = os.path.normpath(str(root))
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, normalized)
            self._roots_list.addItem(item)

    def _set_initial_path(self, initial_path: str) -> None:
        raw = initial_path.strip() if initial_path else ""
        self._pending_select_basename = None
        target_dir = ""
        if raw:
            normalized = os.path.normpath(raw)
            if os.path.isfile(normalized):
                self._pending_select_basename = os.path.basename(normalized)
                target_dir = os.path.dirname(normalized) or normalized
            elif os.path.isdir(normalized):
                target_dir = normalized
            else:
                parent = os.path.dirname(normalized)
                base = os.path.basename(normalized)
                if parent and os.path.isdir(parent):
                    target_dir = parent
                    self._pending_select_basename = base
                else:
                    roots = _DirectoryPickerCache.get_roots()
                    target_dir = roots[0][0] if roots else os.path.expanduser("~")
        if not target_dir:
            roots = _DirectoryPickerCache.get_roots()
            target_dir = roots[0][0] if roots else os.path.expanduser("~")
        self._navigate_to_directory(target_dir)

    def _refresh_roots(self) -> None:
        _DirectoryPickerCache.invalidate_roots()
        self._load_roots()

    def _refresh_current_directory(self) -> None:
        if not self._current_directory:
            return
        _DirectoryPickerCache.invalidate_subdirs(self._current_directory)
        self._populate_subdirs()
        self._populate_files()

    def _create_directory(self) -> None:
        if not self._current_directory:
            return
        folder_name, ok = QInputDialog.getText(
            self,
            _("Create New Folder"),
            _("Folder name:"),
        )
        if not ok:
            return
        folder_name = folder_name.strip()
        if not folder_name:
            self._status_label.setText(_("Folder name cannot be empty."))
            self._status_label.setVisible(True)
            return
        target_path = os.path.normpath(os.path.join(self._current_directory, folder_name))
        try:
            os.makedirs(target_path, exist_ok=False)
            _DirectoryPickerCache.invalidate_subdirs(self._current_directory)
            _DirectoryPickerCache.invalidate_subdirs(target_path)
            self._navigate_to_directory(target_path)
        except FileExistsError:
            self._status_label.setText(_("A folder with that name already exists."))
            self._status_label.setVisible(True)
        except OSError as e:
            logger.warning("Failed to create directory '%s': %s", target_path, e)
            self._status_label.setText(_("Failed to create folder (check permissions/path)."))
            self._status_label.setVisible(True)

    def _on_root_double_clicked(self, item: QListWidgetItem) -> None:
        root = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if root:
            self._navigate_to_directory(root)

    def _on_subdir_double_clicked(self, item: QListWidgetItem) -> None:
        directory = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if directory:
            self._navigate_to_directory(directory)

    def _on_file_double_clicked(self, item: QListWidgetItem) -> None:
        path = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if path and os.path.isfile(path):
            self.selected_file = os.path.normpath(path)
            self.accept()

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            return
        normalized = os.path.normpath(path)
        if os.path.isfile(normalized):
            self._navigate_to_directory(os.path.dirname(normalized) or normalized)
            self._select_file_in_list(os.path.basename(normalized))
            return
        if os.path.isdir(normalized):
            self._navigate_to_directory(normalized)
            return
        self._status_label.setText(_("Path not found or not accessible."))
        self._status_label.setVisible(True)

    def _go_up(self) -> None:
        if not self._current_directory:
            return
        parent = os.path.dirname(os.path.normpath(self._current_directory))
        if not parent:
            return
        self._navigate_to_directory(parent)

    def _navigate_to_directory(self, directory: str) -> None:
        normalized = os.path.normpath(directory.strip()) if directory else ""
        if not normalized:
            return
        self._current_directory = normalized
        self._path_edit.setText(normalized)
        self._populate_subdirs()
        self._populate_files()
        if self._pending_select_basename:
            if self._select_file_in_list(self._pending_select_basename):
                self._pending_select_basename = None

    def _select_file_in_list(self, basename: str) -> bool:
        target = basename.casefold()
        for row in range(self._files_list.count()):
            it = self._files_list.item(row)
            if it is None:
                continue
            if it.text().casefold() == target:
                self._files_list.setCurrentRow(row)
                return True
        return False

    def _populate_subdirs(self) -> None:
        self._subdirs_list.clear()
        self._status_label.setVisible(False)
        children = _DirectoryPickerCache.get_subdirs(self._current_directory)
        for child in children:
            name = os.path.basename(os.path.normpath(child)) or child
            item = QListWidgetItem(name)
            item.setToolTip(child)
            item.setData(Qt.ItemDataRole.UserRole, child)
            self._subdirs_list.addItem(item)
        if not children:
            self._status_label.setText(
                _("No folders found (or folder is not currently accessible).")
            )
            self._status_label.setVisible(True)

    def _populate_files(self) -> None:
        self._files_list.clear()
        rows = list_matching_files(self._current_directory, self._exts)
        for full_path, name in rows:
            item = QListWidgetItem(name)
            item.setToolTip(full_path)
            item.setData(Qt.ItemDataRole.UserRole, full_path)
            self._files_list.addItem(item)
        if self._files_list.count() > 0:
            self._status_label.setVisible(False)

    def _accept_file(self) -> None:
        fi = self._files_list.currentItem()
        if fi is not None:
            path = str(fi.data(Qt.ItemDataRole.UserRole) or "").strip()
            if path and os.path.isfile(path):
                self.selected_file = os.path.normpath(path)
                self.accept()
                return
        sub = self._subdirs_list.currentItem()
        if sub is not None:
            directory = str(sub.data(Qt.ItemDataRole.UserRole) or "").strip()
            if directory:
                self._navigate_to_directory(directory)
                return
        typed = self._path_edit.text().strip()
        if typed:
            n = os.path.normpath(typed)
            if os.path.isfile(n):
                self.selected_file = n
                self.accept()
                return
            if os.path.isdir(n):
                self._navigate_to_directory(n)
                return
        self._status_label.setText(_("Select a file or navigate into a folder."))
        self._status_label.setVisible(True)


class FastSaveFilePickerDialog(QDialog):
    """Efficient save-file picker: navigate like open; filename is typed (no native dialog)."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        title: str,
        initial_path: str = "",
        name_filter: str = "",
        quick_access_locations: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or _("Save File"))
        self.resize(900, 640)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.selected_path = ""
        self._current_directory = ""
        self._exts = parse_qt_name_filter(name_filter)
        self._quick_access_locations = (
            list(quick_access_locations) if quick_access_locations else []
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        content = QHBoxLayout()
        content.setSpacing(8)
        outer.addLayout(content, 1)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        content.addLayout(left_col, 1)

        left_col.addWidget(QLabel(_("Locations")))
        self._roots_list = QListWidget()
        self._roots_list.itemDoubleClicked.connect(self._on_root_double_clicked)
        left_col.addWidget(self._roots_list, 1)

        refresh_roots_btn = QPushButton(_("Refresh locations"))
        refresh_roots_btn.clicked.connect(self._refresh_roots)
        left_col.addWidget(refresh_roots_btn)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        content.addLayout(right_col, 3)

        path_bar = QHBoxLayout()
        path_bar.setSpacing(6)
        right_col.addLayout(path_bar)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(_("Type or paste a directory path"))
        self._path_edit.returnPressed.connect(self._go_to_path)
        path_bar.addWidget(self._path_edit, 1)

        go_btn = QPushButton(_("Go"))
        go_btn.clicked.connect(self._go_to_path)
        path_bar.addWidget(go_btn)

        up_btn = QPushButton(_("Up"))
        up_btn.clicked.connect(self._go_up)
        path_bar.addWidget(up_btn)

        new_dir_btn = QPushButton(_("New Folder"))
        new_dir_btn.clicked.connect(self._create_directory)
        path_bar.addWidget(new_dir_btn)

        refresh_btn = QPushButton(_("Refresh"))
        refresh_btn.clicked.connect(self._refresh_current_directory)
        path_bar.addWidget(refresh_btn)

        right_col.addWidget(QLabel(_("Folders")))
        self._subdirs_list = QListWidget()
        self._subdirs_list.itemDoubleClicked.connect(self._on_subdir_double_clicked)
        right_col.addWidget(self._subdirs_list, 1)

        right_col.addWidget(QLabel(_("Files (optional — double-click to fill name)")))
        self._files_list = QListWidget()
        self._files_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        right_col.addWidget(self._files_list, 1)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(QLabel(_("File name")))
        self._filename_edit = QLineEdit()
        self._filename_edit.setPlaceholderText(_("output.onnx"))
        name_row.addWidget(self._filename_edit, 1)
        right_col.addLayout(name_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #ff7777;")
        self._status_label.setVisible(False)
        right_col.addWidget(self._status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        outer.addLayout(buttons)

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        save_btn = QPushButton(_("Save"))
        save_btn.clicked.connect(self._accept_save)
        save_btn.setDefault(True)
        buttons.addWidget(save_btn)

        self._load_roots()
        self._set_initial_path(initial_path)
        _center_dialog_on_parent_or_screen(self, parent)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            self.reject()
            return
        super().keyPressEvent(event)

    def _load_roots(self) -> None:
        self._roots_list.clear()
        seen: set[str] = set()
        all_locations = list(self._quick_access_locations) + _DirectoryPickerCache.get_roots()
        for root, label in all_locations:
            normalized = os.path.normpath(str(root))
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, normalized)
            self._roots_list.addItem(item)

    def _set_initial_path(self, initial_path: str) -> None:
        raw = initial_path.strip() if initial_path else ""
        default_name = ""
        target_dir = ""
        if raw:
            normalized = os.path.normpath(raw)
            if os.path.isfile(normalized):
                default_name = os.path.basename(normalized)
                target_dir = os.path.dirname(normalized) or normalized
            elif os.path.isdir(normalized):
                target_dir = normalized
            else:
                parent = os.path.dirname(normalized)
                base = os.path.basename(normalized)
                if parent and os.path.isdir(parent):
                    target_dir = parent
                    default_name = base
                else:
                    roots = _DirectoryPickerCache.get_roots()
                    target_dir = roots[0][0] if roots else os.path.expanduser("~")
        if not target_dir:
            roots = _DirectoryPickerCache.get_roots()
            target_dir = roots[0][0] if roots else os.path.expanduser("~")
        self._filename_edit.setText(default_name)
        self._navigate_to_directory(target_dir)

    def _refresh_roots(self) -> None:
        _DirectoryPickerCache.invalidate_roots()
        self._load_roots()

    def _refresh_current_directory(self) -> None:
        if not self._current_directory:
            return
        _DirectoryPickerCache.invalidate_subdirs(self._current_directory)
        self._populate_subdirs()
        self._populate_files()

    def _create_directory(self) -> None:
        if not self._current_directory:
            return
        folder_name, ok = QInputDialog.getText(
            self,
            _("Create New Folder"),
            _("Folder name:"),
        )
        if not ok:
            return
        folder_name = folder_name.strip()
        if not folder_name:
            self._status_label.setText(_("Folder name cannot be empty."))
            self._status_label.setVisible(True)
            return
        target_path = os.path.normpath(os.path.join(self._current_directory, folder_name))
        try:
            os.makedirs(target_path, exist_ok=False)
            _DirectoryPickerCache.invalidate_subdirs(self._current_directory)
            _DirectoryPickerCache.invalidate_subdirs(target_path)
            self._navigate_to_directory(target_path)
        except FileExistsError:
            self._status_label.setText(_("A folder with that name already exists."))
            self._status_label.setVisible(True)
        except OSError as e:
            logger.warning("Failed to create directory '%s': %s", target_path, e)
            self._status_label.setText(_("Failed to create folder (check permissions/path)."))
            self._status_label.setVisible(True)

    def _on_root_double_clicked(self, item: QListWidgetItem) -> None:
        root = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if root:
            self._navigate_to_directory(root)

    def _on_subdir_double_clicked(self, item: QListWidgetItem) -> None:
        directory = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if directory:
            self._navigate_to_directory(directory)

    def _on_file_double_clicked(self, item: QListWidgetItem) -> None:
        name = item.text()
        if name:
            self._filename_edit.setText(name)

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            return
        normalized = os.path.normpath(path)
        if os.path.isfile(normalized):
            self._filename_edit.setText(os.path.basename(normalized))
            self._navigate_to_directory(os.path.dirname(normalized) or normalized)
            return
        if os.path.isdir(normalized):
            self._navigate_to_directory(normalized)
            return
        self._status_label.setText(_("Path not found or not accessible."))
        self._status_label.setVisible(True)

    def _go_up(self) -> None:
        if not self._current_directory:
            return
        parent = os.path.dirname(os.path.normpath(self._current_directory))
        if not parent:
            return
        self._navigate_to_directory(parent)

    def _navigate_to_directory(self, directory: str) -> None:
        normalized = os.path.normpath(directory.strip()) if directory else ""
        if not normalized:
            return
        self._current_directory = normalized
        self._path_edit.setText(normalized)
        self._populate_subdirs()
        self._populate_files()

    def _populate_subdirs(self) -> None:
        self._subdirs_list.clear()
        self._status_label.setVisible(False)
        children = _DirectoryPickerCache.get_subdirs(self._current_directory)
        for child in children:
            name = os.path.basename(os.path.normpath(child)) or child
            item = QListWidgetItem(name)
            item.setToolTip(child)
            item.setData(Qt.ItemDataRole.UserRole, child)
            self._subdirs_list.addItem(item)
        if not children:
            self._status_label.setText(
                _("No folders found (or folder is not currently accessible).")
            )
            self._status_label.setVisible(True)

    def _populate_files(self) -> None:
        self._files_list.clear()
        rows = list_matching_files(self._current_directory, self._exts)
        for full_path, name in rows:
            item = QListWidgetItem(name)
            item.setToolTip(full_path)
            item.setData(Qt.ItemDataRole.UserRole, full_path)
            self._files_list.addItem(item)
        if self._files_list.count() > 0:
            self._status_label.setVisible(False)

    def _accept_save(self) -> None:
        name = self._filename_edit.text().strip()
        if not name:
            self._status_label.setText(_("Enter a file name."))
            self._status_label.setVisible(True)
            return
        if os.path.sep in name or (os.altsep and os.altsep in name):
            self._status_label.setText(_("Use a simple file name (no path separators)."))
            self._status_label.setVisible(True)
            return
        full = os.path.normpath(os.path.join(self._current_directory, name))
        self.selected_path = full
        self.accept()


def _merge_quick_access(
    parent: QWidget | None,
    quick_access_locations: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    resolved = list(quick_access_locations) if quick_access_locations else []
    if parent is not None and hasattr(parent, "get_base_dir"):
        try:
            base_dir = parent.get_base_dir()
            if isinstance(base_dir, str) and base_dir.strip() != "" and os.path.isdir(base_dir):
                resolved.append(
                    (os.path.normpath(base_dir), _("Current Base Directory"))
                )
        except Exception:
            pass
    return resolved


def get_existing_directory(
    parent: QWidget | None,
    title: str,
    initial_dir: str = "",
    quick_access_locations: list[tuple[str, str]] | None = None,
) -> str:
    """
    Drop-in replacement for QFileDialog.getExistingDirectory.

    Returns:
        Selected directory path, or empty string if cancelled.
    """
    global _session_last_directory_hint
    global _session_has_opened_picker

    effective_initial_dir = (initial_dir or "").strip()
    if effective_initial_dir:
        _session_last_directory_hint = effective_initial_dir
    elif _session_last_directory_hint:
        # Reuse last known context directory when caller doesn't provide one.
        effective_initial_dir = _session_last_directory_hint
    elif not _session_has_opened_picker:
        # First open in process: use current working directory as broad context hint.
        try:
            cwd = os.getcwd()
            if cwd and os.path.isdir(cwd):
                effective_initial_dir = cwd
        except OSError:
            pass

    resolved_quick_access = _merge_quick_access(parent, quick_access_locations)

    dialog = FastDirectoryPickerDialog(
        parent,
        title=title or _("Select Directory"),
        initial_dir=effective_initial_dir,
        quick_access_locations=resolved_quick_access,
    )
    _session_has_opened_picker = True
    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected = dialog.selected_directory or ""
        if selected:
            _session_last_directory_hint = selected
        return selected
    return ""


def get_open_file_name(
    parent: QWidget | None,
    title: str,
    initial_path: str = "",
    name_filter: str = "",
    quick_access_locations: list[tuple[str, str]] | None = None,
) -> str:
    """
    Drop-in replacement for ``QFileDialog.getOpenFileName`` (returns path only).

    Only lists files in the directory currently shown; filters apply to that list.
    """
    global _session_last_directory_hint
    global _session_has_opened_picker

    effective = (initial_path or "").strip()
    if effective:
        if os.path.isfile(effective):
            _session_last_directory_hint = os.path.dirname(effective) or effective
        elif os.path.isdir(effective):
            _session_last_directory_hint = effective
        else:
            _session_last_directory_hint = effective
    elif _session_last_directory_hint:
        effective = _session_last_directory_hint
    elif not _session_has_opened_picker:
        try:
            cwd = os.getcwd()
            if cwd and os.path.isdir(cwd):
                effective = cwd
        except OSError:
            pass

    dialog = FastOpenFilePickerDialog(
        parent,
        title=title or _("Open File"),
        initial_path=effective,
        name_filter=name_filter,
        quick_access_locations=_merge_quick_access(parent, quick_access_locations),
    )
    _session_has_opened_picker = True
    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected = dialog.selected_file or ""
        if selected:
            parent_dir = os.path.dirname(selected)
            _session_last_directory_hint = parent_dir or selected
        return selected
    return ""


def get_save_file_name(
    parent: QWidget | None,
    title: str,
    initial_path: str = "",
    name_filter: str = "",
    quick_access_locations: list[tuple[str, str]] | None = None,
) -> str:
    """
    Drop-in replacement for ``QFileDialog.getSaveFileName`` (returns path only).

    User picks a directory in the fast navigator and types the file name below.
    """
    global _session_last_directory_hint
    global _session_has_opened_picker

    effective = (initial_path or "").strip()
    if effective:
        if os.path.isfile(effective):
            _session_last_directory_hint = os.path.dirname(effective) or effective
        elif os.path.isdir(effective):
            _session_last_directory_hint = effective
        else:
            _session_last_directory_hint = effective
    elif _session_last_directory_hint:
        effective = _session_last_directory_hint
    elif not _session_has_opened_picker:
        try:
            cwd = os.getcwd()
            if cwd and os.path.isdir(cwd):
                effective = cwd
        except OSError:
            pass

    dialog = FastSaveFilePickerDialog(
        parent,
        title=title or _("Save File"),
        initial_path=effective,
        name_filter=name_filter,
        quick_access_locations=_merge_quick_access(parent, quick_access_locations),
    )
    _session_has_opened_picker = True
    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected = dialog.selected_path or ""
        if selected:
            parent_dir = os.path.dirname(selected)
            _session_last_directory_hint = parent_dir or selected
        return selected
    return ""
