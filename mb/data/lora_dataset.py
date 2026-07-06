"""
LoRA-branch data prep for ``mb data create-dataset --model-type image_generation_lora``.

Unlike :class:`~mb.data.dataset.DatasetCreator` (ImageFolder class-folder train/test split
for image classification), a LoRA training set for
:attr:`~mb.models.types.ModelType.IMAGE_GENERATION_LORA` is a single flat directory of
images, each with an optional paired caption (see :mod:`mb.data.lora_captions`) — there are
no classes and no held-out test split; LoRA fine-tuning typically trains on the whole
curated set. ``--raw-data-dir`` is expected to contain images **directly** (not nested
under per-class ``CONVERTED/`` folders like the classification path).

This is deliberately a standalone module, not a branch inside the shared per-class-folder
helpers in :mod:`mb.data.dataset` / :mod:`mb.data.class_layout` — the data shape and copy
semantics differ too much from the class-folder pipeline to share that code safely without
risking the existing image-classification flows.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mb.cancellation import check_cancel_event
from mb.data.file_types import configured_media_suffixes
from mb.data.lora_captions import read_caption, write_caption
from mb.data.media_utils import pil_image_to_jpeg_normalized
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)


@dataclass
class LoraDatasetImage:
    """One prepared training image, relative to ``data_dir``."""

    output_path: str
    source_path: str
    caption: Optional[str]


@dataclass
class LoraDatasetReport:
    """Result of :meth:`LoraDatasetCreator.run`."""

    raw_data_dir: Path
    data_dir: Path
    n_scanned: int = 0
    n_copied: int = 0
    n_captioned: int = 0
    n_skipped: int = 0
    images: List[LoraDatasetImage] = field(default_factory=list)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "raw_data_dir": str(self.raw_data_dir),
            "data_dir": str(self.data_dir),
            "n_scanned": self.n_scanned,
            "n_copied": self.n_copied,
            "n_captioned": self.n_captioned,
            "n_skipped": self.n_skipped,
            "images": [
                {"output_path": i.output_path, "source_path": i.source_path, "caption": i.caption}
                for i in self.images
            ],
        }


class LoraDatasetCreator:
    """
    Copy/normalize a flat folder of images (+ optional ``.txt`` caption sidecars) into a
    LoRA training directory.

    No class folders, no train/test split — every valid image under ``raw_data_dir`` is
    copied to ``data_dir``. Output files are named by content hash (SHA-256 of the
    *source* file bytes) to avoid collisions, matching the collision-safe convention used
    elsewhere in the data pipeline (:class:`~mb.data.dataset.DatasetCreator`); each image's
    caption (if any) is written alongside it with the same stem.
    """

    def __init__(self, raw_data_dir: Path, data_dir: Path) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.data_dir = Path(data_dir)
        self.report: Optional[LoraDatasetReport] = None
        self._cancel_event: Optional[threading.Event] = None

    def _scan_source_images(self) -> List[Path]:
        exts = configured_media_suffixes()
        return sorted(
            p for p in self.raw_data_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
        )

    def run(self, cancel_event: Optional[threading.Event] = None) -> bool:
        """Main execution method. *cancel_event* is checked periodically during the copy loop."""
        self._cancel_event = cancel_event
        try:
            return self._run_impl()
        finally:
            self._cancel_event = None

    def _run_impl(self) -> bool:
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Output data directory: {self.data_dir}")

        check_cancel_event(self._cancel_event)

        if not self.raw_data_dir.is_dir():
            logger.error(_("Raw data directory does not exist: {path}").format(path=self.raw_data_dir))
            return False

        images = self._scan_source_images()
        n_scanned = len(images)
        if n_scanned == 0:
            logger.error(_("No images found under {path}").format(path=self.raw_data_dir))
            return False

        self.data_dir.mkdir(parents=True, exist_ok=True)

        prepared: List[LoraDatasetImage] = []
        n_copied = 0
        n_captioned = 0
        n_skipped = 0

        for i, source_path in enumerate(images, 1):
            if i % 1000 == 0:
                check_cancel_event(self._cancel_event)
                logger.info(f"Preparing LoRA dataset: {i}/{n_scanned} images")

            target_path = self._copy_one_image(source_path)
            if target_path is None:
                n_skipped += 1
                continue

            caption = read_caption(source_path)
            if caption:
                write_caption(target_path, caption)
                n_captioned += 1

            prepared.append(
                LoraDatasetImage(
                    output_path=target_path.name,
                    source_path=str(source_path),
                    caption=caption,
                )
            )
            n_copied += 1

        logger.info(
            f"LoRA dataset prepared: {n_copied}/{n_scanned} images copied to {self.data_dir} "
            f"({n_captioned} with captions, {n_skipped} skipped)"
        )

        self.report = LoraDatasetReport(
            raw_data_dir=self.raw_data_dir,
            data_dir=self.data_dir,
            n_scanned=n_scanned,
            n_copied=n_copied,
            n_captioned=n_captioned,
            n_skipped=n_skipped,
            images=prepared,
        )
        return n_copied > 0

    def _copy_one_image(self, source_path: Path) -> Optional[Path]:
        """Copy/normalize *source_path* into ``data_dir``; return the target path, or ``None`` on failure."""
        try:
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError as e:
            logger.warning(f"Could not read {source_path}: {e}")
            return None
        target_path = self.data_dir / f"{digest}.jpg"

        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        if source_path.suffix.lower() in (".jpg", ".jpeg"):
            try:
                shutil.copy2(source_path, target_path)
                ok = target_path.exists() and target_path.stat().st_size > 0
            except OSError as e:
                logger.warning(f"Could not copy {source_path}: {e}")
                ok = False
            return target_path if ok else None

        try:
            from PIL import Image

            with Image.open(source_path) as img:
                ok = pil_image_to_jpeg_normalized(img, target_path)
        except OSError as e:
            logger.warning(f"Could not open {source_path}: {e}")
            ok = False
        return target_path if ok else None
