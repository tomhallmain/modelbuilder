#!/usr/bin/env python3
"""
One-off cleanup script for paired dataset files.

Input: text file containing one absolute path per line, expected shape:
    {abs_path}\\raw_data\\{class_dir_name}\\IMAGES\\filepath.{ext}

Behavior:
    - For each listed IMAGES file, derive sibling CONVERTED path:
      {abs_path}\\raw_data\\{class_dir_name}\\CONVERTED\\filepath.{ext}
    - If CONVERTED exists, remove both IMAGES and CONVERTED.
    - If CONVERTED does not exist, remove IMAGES only.
    - Dry-run by default; pass --apply to perform deletions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove IMAGES files listed in a text file and matching CONVERTED siblings."
    )
    parser.add_argument(
        "list_file",
        type=Path,
        help="Path to text file containing absolute IMAGES paths, one per line.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script performs a dry run.",
    )
    return parser.parse_args()


def iter_nonempty_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def is_expected_images_path(file_path: Path) -> bool:
    # Expected suffix: .../raw_data/<class_dir_name>/IMAGES/<filename>
    if len(file_path.parts) < 4:
        return False
    return (
        file_path.parts[-4].lower() == "raw_data"
        and file_path.parts[-2].upper() == "IMAGES"
    )


def mapped_converted_path(images_path: Path) -> Path:
    # Replace only the parent directory name IMAGES -> CONVERTED
    return images_path.parent.parent / "CONVERTED" / images_path.name


def delete_if_exists(path: Path, apply: bool) -> bool:
    if not path.exists():
        return False
    if apply:
        path.unlink()
    return True


def main() -> int:
    args = parse_args()
    list_file: Path = args.list_file
    apply: bool = args.apply

    if not list_file.exists():
        print(f"ERROR: list file does not exist: {list_file}")
        return 2
    if not list_file.is_file():
        print(f"ERROR: list path is not a file: {list_file}")
        return 2

    total_lines = 0
    invalid_paths = 0
    missing_images = 0
    images_deleted_or_would_delete = 0
    converted_deleted_or_would_delete = 0

    mode_label = "APPLY" if apply else "DRY RUN"
    print(f"[{mode_label}] processing list: {list_file}")

    for line in iter_nonempty_lines(list_file):
        total_lines += 1
        images_path = Path(line)

        if not images_path.is_absolute() or not is_expected_images_path(images_path):
            invalid_paths += 1
            print(f"SKIP invalid format: {images_path}")
            continue

        converted_path = mapped_converted_path(images_path)

        images_exists = images_path.exists()
        converted_exists = converted_path.exists()

        if not images_exists:
            missing_images += 1
            if converted_exists:
                action = "DELETE" if apply else "WOULD DELETE"
                print(
                    f"{action} converted only (images missing): {converted_path}"
                )
                if delete_if_exists(converted_path, apply):
                    converted_deleted_or_would_delete += 1
            else:
                print(f"SKIP missing images and no converted: {images_path}")
            continue

        action = "DELETE" if apply else "WOULD DELETE"
        print(f"{action} images: {images_path}")
        if delete_if_exists(images_path, apply):
            images_deleted_or_would_delete += 1

        if converted_exists:
            print(f"{action} converted: {converted_path}")
            if delete_if_exists(converted_path, apply):
                converted_deleted_or_would_delete += 1
        else:
            print(f"INFO no converted sibling: {converted_path}")

    print("\n--- Summary ---")
    print(f"Mode: {mode_label}")
    print(f"Input entries processed: {total_lines}")
    print(f"Invalid path entries skipped: {invalid_paths}")
    print(f"Missing images entries: {missing_images}")
    print(
        f"IMAGES {'deleted' if apply else 'to delete'}: {images_deleted_or_would_delete}"
    )
    print(
        f"CONVERTED {'deleted' if apply else 'to delete'}: {converted_deleted_or_would_delete}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
