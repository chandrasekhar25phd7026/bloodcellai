"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    utils.py

Version:
    2.0.0

Purpose:
    Utility functions shared across the validation engine.

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

# =============================================================================
# Supported Extensions
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

ANNOTATION_EXTENSIONS = {
    ".txt",
    ".xml",
    ".json",
}

# =============================================================================
# Path Utilities
# =============================================================================

def normalize_path(path: str | Path) -> Path:
    """
    Convert a path into an absolute resolved Path object.
    """
    return Path(path).expanduser().resolve()


def path_exists(path: str | Path) -> bool:
    """
    Returns True if the path exists.
    """
    return normalize_path(path).exists()


def is_file(path: str | Path) -> bool:
    """
    Returns True if the path is a file.
    """
    return normalize_path(path).is_file()


def is_directory(path: str | Path) -> bool:
    """
    Returns True if the path is a directory.
    """
    return normalize_path(path).is_dir()


# =============================================================================
# Extension Utilities
# =============================================================================

def get_extension(path: str | Path) -> str:
    """
    Returns the lowercase file extension.
    """
    return normalize_path(path).suffix.lower()


def is_image_file(path: str | Path) -> bool:
    """
    Check whether a file is a supported image.
    """
    return get_extension(path) in IMAGE_EXTENSIONS


def is_annotation_file(path: str | Path) -> bool:
    """
    Check whether a file is a supported annotation.
    """
    return get_extension(path) in ANNOTATION_EXTENSIONS
# =============================================================================
# File Information
# =============================================================================

def get_filename(path: str | Path) ->str:
    """
    Return filename with extension.
    """
    return normalize_path(path).name


def get_stem(path: str | Path) -> str:
    """
    Return filename without extension.
    """
    return normalize_path(path).stem


def get_file_size(path: str | Path) -> int:
    """
    Return file size in bytes.
    Returns 0 if the file does not exist.
    """
    file_path = normalize_path(path)

    if not file_path.exists():
        return 0

    return file_path.stat().st_size


# =============================================================================
# File Hash Utilities
# =============================================================================

def compute_sha256(
    path: str | Path,
    chunk_size: int = 65536,
) -> str:
    """
    Compute SHA256 hash of a file.

    Parameters
    ----------
    path : str | Path
        Input file.

    chunk_size : int
        Read block size.

    Returns
    -------
    str
        SHA256 hexadecimal digest.
    """

    file_path = normalize_path(path)

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(chunk_size)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# =============================================================================
# Duplicate Utilities
# =============================================================================

def is_duplicate_filename(
    filename: str,
    seen_filenames: set[str],
) -> bool:
    """
    Detect duplicate filenames.
    """

    if filename in seen_filenames:
        return True

    seen_filenames.add(filename)

    return False


def is_duplicate_path(
    path: str | Path,
    seen_paths: set[Path],
) -> bool:
    """
    Detect duplicate image paths.
    """

    file_path = normalize_path(path)

    if file_path in seen_paths:
        return True

    seen_paths.add(file_path)

    return False


def is_duplicate_hash(
    file_hash: str,
    seen_hashes: set[str],
) -> bool:
    """
    Detect duplicate file hashes.
    """

    if file_hash in seen_hashes:
        return True

    seen_hashes.add(file_hash)

    return False


# =============================================================================
# Collection Utilities
# =============================================================================

def unique_items(items: Iterable) -> list:
    """
    Return unique items while preserving order.
    """

    seen = set()

    output = []

    for item in items:

        if item not in seen:

            seen.add(item)

            output.append(item)

    return output


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe division.

    Returns 0.0 when denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator
# =============================================================================
# Image Utilities
# =============================================================================

def get_image_files(
    directory: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """
    Return all supported image files.

    Parameters
    ----------
    directory : str | Path
        Dataset directory.

    recursive : bool, default=True
        If True, search all subdirectories recursively.

    Returns
    -------
    list[Path]
        Sorted list of image files.
    """

    directory = normalize_path(directory)

    if not directory.exists():
        return []

    if not directory.is_dir():
        return []

    if recursive:

        image_files = [
            file
            for file in directory.rglob("*")
            if file.is_file()
            and is_image_file(file)
        ]

    else:

        image_files = [
            file
            for file in directory.iterdir()
            if file.is_file()
            and is_image_file(file)
        ]

    return sorted(image_files)


def get_annotation_files(
    directory: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """
    Return all supported annotation files.

    Parameters
    ----------
    directory : str | Path
        Dataset directory.

    recursive : bool, default=True
        If True, search all subdirectories recursively.

    Returns
    -------
    list[Path]
        Sorted list of annotation files.
    """

    directory = normalize_path(directory)

    if not directory.exists():
        return []

    if not directory.is_dir():
        return []

    if recursive:

        annotation_files = [
            file
            for file in directory.rglob("*")
            if file.is_file()
            and is_annotation_file(file)
        ]

    else:

        annotation_files = [
            file
            for file in directory.iterdir()
            if file.is_file()
            and is_annotation_file(file)
        ]

    return sorted(annotation_files)
def find_dataset_splits(
    directory: str | Path,
) -> dict[str, Path]:
    """
    Automatically detect train/valid/test directories.

    Returns
    -------
    {
        "train": Path(...),
        "valid": Path(...),
        "test": Path(...)
    }
    """

    directory = normalize_path(directory)

    splits = {}

    for folder in directory.rglob("*"):

        if not folder.is_dir():
            continue

        name = folder.name.lower()

        if name in (
            "train",
            "training",
        ):
            splits["train"] = folder

        elif name in (
            "valid",
            "validation",
            "val",
        ):
            splits["valid"] = folder

        elif name == "test":
            splits["test"] = folder

    return splits


# =============================================================================
# Dataset Structure Utilities
# =============================================================================

def find_dataset_splits(
    directory: str | Path,
) -> dict[str, Path]:
    """
    Automatically detect train/validation/test folders.

    Supported names
    ---------------
    train, training
    valid, validation, val
    test

    Returns
    -------
    dict[str, Path]
    """

    directory = normalize_path(directory)

    splits: dict[str, Path] = {}

    if not directory.exists():
        return splits

    for folder in directory.rglob("*"):

        if not folder.is_dir():
            continue

        name = folder.name.lower()

        if name in ("train", "training"):
            splits["train"] = folder

        elif name in ("valid", "validation", "val"):
            splits["valid"] = folder

        elif name == "test":
            splits["test"] = folder

    return splits

# =============================================================================
# Directory Utilities
# =============================================================================

def list_files(
    directory: str | Path,
    recursive: bool = False,
) -> list[Path]:
    """
    List all files in a directory.
    """

    directory = normalize_path(directory)

    if not directory.exists():
        return []

    if recursive:
        return sorted(
            [
                file
                for file in directory.rglob("*")
                if file.is_file()
            ]
        )

    return sorted(
        [
            file
            for file in directory.iterdir()
            if file.is_file()
        ]
    )


def count_files(
    directory: str | Path,
    recursive: bool = False,
) -> int:
    """
    Count files in a directory.
    """

    return len(
        list_files(
            directory,
            recursive=recursive,
        )
    )


# =============================================================================
# Dataset Utilities
# =============================================================================

def count_images(directory: str | Path) -> int:
    """
    Count image files in a directory.
    """

    return len(get_image_files(directory))


def count_annotations(directory: str | Path) -> int:
    """
    Count annotation files in a directory.
    """

    return len(get_annotation_files(directory))


def has_images(directory: str | Path) -> bool:
    """
    True if directory contains images.
    """

    return count_images(directory) > 0


def has_annotations(directory: str | Path) -> bool:
    """
    True if directory contains annotations.
    """

    return count_annotations(directory) > 0


# =============================================================================
# Validation Helpers
# =============================================================================

def is_empty(value) -> bool:
    """
    Check whether a value is empty.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    try:
        return len(value) == 0
    except TypeError:
        return False


def ensure_list(value) -> list:
    """
    Convert a value to a list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]
# =============================================================================
# Optional Image Loading
# =============================================================================

try:
    from PIL import Image
except ImportError:
    Image = None


def get_image_size(path: str | Path) -> tuple[int, int]:
    """
    Return (width, height) of an image.

    Returns
    -------
    tuple
        (0, 0) if Pillow is unavailable or the image
        cannot be opened.
    """

    if Image is None:
        return (0, 0)

    try:

        with Image.open(normalize_path(path)) as img:

            return img.size

    except Exception:

        return (0, 0)


def is_valid_image(path: str | Path) -> bool:
    """
    Verify that an image can be opened successfully.
    """

    if Image is None:
        return False

    try:

        with Image.open(normalize_path(path)) as img:
            img.verify()

        return True

    except Exception:

        return False


# =============================================================================
# Public API
# =============================================================================

__all__ = [

    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "IMAGE_EXTENSIONS",
    "ANNOTATION_EXTENSIONS",

    # -------------------------------------------------------------------------
    # Path Utilities
    # -------------------------------------------------------------------------
    "normalize_path",
    "path_exists",
    "is_file",
    "is_directory",

    # -------------------------------------------------------------------------
    # Extension Utilities
    # -------------------------------------------------------------------------
    "get_extension",
    "is_image_file",
    "is_annotation_file",

    # -------------------------------------------------------------------------
    # File Utilities
    # -------------------------------------------------------------------------
    "get_filename",
    "get_stem",
    "get_file_size",

    # -------------------------------------------------------------------------
    # Hash Utilities
    # -------------------------------------------------------------------------
    "compute_sha256",

    # -------------------------------------------------------------------------
    # Duplicate Utilities
    # -------------------------------------------------------------------------
    "is_duplicate_filename",
    "is_duplicate_path",
    "is_duplicate_hash",

    # -------------------------------------------------------------------------
    # Collection Utilities
    # -------------------------------------------------------------------------
    "unique_items",
    "safe_divide",

    # -------------------------------------------------------------------------
    # Image Utilities
    # -------------------------------------------------------------------------
    "get_image_files",
    "get_annotation_files",
    "find_dataset_splits",

    # -------------------------------------------------------------------------
    # Directory Utilities
    # -------------------------------------------------------------------------
    "list_files",
    "count_files",

    # -------------------------------------------------------------------------
    # Dataset Utilities
    # -------------------------------------------------------------------------
    "count_images",
    "count_annotations",
    "has_images",
    "has_annotations",

    # -------------------------------------------------------------------------
    # Validation Helpers
    # -------------------------------------------------------------------------
    "is_empty",
    "ensure_list",

    # -------------------------------------------------------------------------
    # Image Validation Utilities
    # -------------------------------------------------------------------------
    "get_image_size",
    "is_valid_image",
]