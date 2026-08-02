"""
BloodCellAI Framework
Validation Utilities

File:
    utils.py

Description
-----------
Common utility functions used throughout the validation package.

Version:
    1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# =============================================================================
# Constants
# =============================================================================

EPSILON = 1e-8

SUPPORTED_IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# =============================================================================
# Safe Attribute Access
# =============================================================================

def safe_get(obj: Any, attribute: str, default=None):
    """
    Safely retrieve an attribute from an object.

    Parameters
    ----------
    obj : Any
    attribute : str
    default : Any

    Returns
    -------
    Any
    """
    return getattr(obj, attribute, default)


# =============================================================================
# Numeric Validation
# =============================================================================

def is_positive(value) -> bool:
    """
    Check whether a value is greater than zero.
    """
    return value is not None and value > 0


def is_non_negative(value) -> bool:
    """
    Check whether a value is >= 0.
    """
    return value is not None and value >= 0


def is_between(value, minimum, maximum) -> bool:
    """
    Check whether a value lies within [minimum, maximum].
    """
    if value is None:
        return False

    return minimum <= value <= maximum


def is_normalized(value) -> bool:
    """
    Check whether a value lies within the YOLO normalized range [0,1].
    """
    return is_between(value, 0.0, 1.0)


# =============================================================================
# Safe Division
# =============================================================================

def safe_divide(numerator, denominator):
    """
    Prevent divide-by-zero errors.
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


# =============================================================================
# File Utilities
# =============================================================================

def get_filename(path: str) -> str:
    """
    Return filename from a path.
    """
    return Path(path).name


def get_stem(path: str) -> str:
    """
    Return filename without extension.
    """
    return Path(path).stem


def get_extension(path: str) -> str:
    """
    Return lowercase file extension.
    """
    return Path(path).suffix.lower()


def file_exists(path: str) -> bool:
    """
    Check whether a file exists.
    """
    return Path(path).exists()


def is_supported_image(path: str) -> bool:
    """
    Check whether an image has a supported extension.
    """
    return get_extension(path) in SUPPORTED_IMAGE_FORMATS


# =============================================================================
# Bounding Box Utilities
# =============================================================================

def bbox_area(width, height):
    """
    Compute bounding box area.
    """
    if width is None or height is None:
        return 0.0

    return width * height


def bbox_is_valid(xc, yc, w, h):
    """
    Validate normalized YOLO bounding box.
    """
    return (
        is_normalized(xc)
        and is_normalized(yc)
        and is_positive(w)
        and is_positive(h)
        and w <= 1
        and h <= 1
    )


# =============================================================================
# Image Utilities
# =============================================================================

def image_has_valid_size(width, height):
    """
    Validate image dimensions.
    """
    return is_positive(width) and is_positive(height)


# =============================================================================
# Statistics Helpers
# =============================================================================

def percentage(part, total):
    """
    Calculate percentage safely.
    """
    return safe_divide(part * 100.0, total)


def average(total, count):
    """
    Compute average safely.
    """
    return safe_divide(total, count)