"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    image_rules.py

Version:
    2.0.0

Purpose:
    Image-level validation rules.

Description:
    This module validates image files before annotation and object
    validation begins.

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

from .models import (
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

from .utils import (
    get_extension,
    get_file_size,
    get_image_size,
    is_image_file,
    is_valid_image,
    normalize_path,
    path_exists,
)

# =============================================================================
# Image Existence Rules
# =============================================================================


def check_image_exists(
    image_path: str | Path,
    dataset: str,
    result: ValidationResult,
) -> None:
    """
    Verify that an image exists.
    """

    image_path = normalize_path(image_path)

    if not path_exists(image_path):

        result.add_issue(
            ValidationIssue(
                issue_id="IMG001",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.IMAGE,
                rule_name="Image Exists",
                dataset=dataset,
                image_path=str(image_path),
                message="Image file does not exist.",
                recommendation="Provide a valid image file.",
            )
        )


# =============================================================================
# Image Extension Rules
# =============================================================================


def check_image_extension(
    image_path: str | Path,
    dataset: str,
    result: ValidationResult,
) -> None:
    """
    Validate image extension.
    """

    image_path = normalize_path(image_path)

    if not is_image_file(image_path):

        result.add_issue(
            ValidationIssue(
                issue_id="IMG002",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.IMAGE,
                rule_name="Image Extension",
                dataset=dataset,
                image_path=str(image_path),
                message=f"Unsupported image extension '{get_extension(image_path)}'.",
                recommendation="Use a supported image format.",
            )
        )
# =============================================================================
# Image Integrity Rules
# =============================================================================

def check_image_corruption(
    image_path: str | Path,
    dataset: str,
    result: ValidationResult,
) -> None:
    """
    Verify that an image is readable.
    """

    image_path = normalize_path(image_path)

    if not path_exists(image_path):
        return

    if not is_valid_image(image_path):

        result.add_issue(
            ValidationIssue(
                issue_id="IMG003",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.IMAGE,
                rule_name="Image Corruption",
                dataset=dataset,
                image_path=str(image_path),
                message="Image is corrupted or unreadable.",
                recommendation="Replace or repair the image.",
            )
        )


def check_empty_image(
    image_path: str | Path,
    dataset: str,
    result: ValidationResult,
) -> None:
    """
    Detect zero-byte image files.
    """

    image_path = normalize_path(image_path)

    if not path_exists(image_path):
        return

    if get_file_size(image_path) == 0:

        result.add_issue(
            ValidationIssue(
                issue_id="IMG004",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.IMAGE,
                rule_name="Empty Image",
                dataset=dataset,
                image_path=str(image_path),
                message="Image file is empty (0 bytes).",
                recommendation="Replace the empty image.",
            )
        )


# =============================================================================
# Image Dimension Rules
# =============================================================================

def check_image_dimensions(
    image_path: str | Path,
    dataset: str,
    result: ValidationResult,
    min_width: int = 32,
    min_height: int = 32,
    max_width: int = 10000,
    max_height: int = 10000,
) -> None:
    """
    Validate image dimensions.
    """

    image_path = normalize_path(image_path)

    if not path_exists(image_path):
        return

    width, height = get_image_size(image_path)

    if width == 0 or height == 0:
        return

    if width < min_width or height < min_height:

        result.add_issue(
            ValidationIssue(
                issue_id="IMG005",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.IMAGE,
                rule_name="Minimum Image Size",
                dataset=dataset,
                image_path=str(image_path),
                message=(
                    f"Image resolution ({width} × {height}) "
                    f"is smaller than the recommended "
                    f"minimum ({min_width} × {min_height})."
                ),
                recommendation="Use higher-resolution images.",
            )
        )

    if width > max_width or height > max_height:

        result.add_issue(
            ValidationIssue(
                issue_id="IMG006",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.IMAGE,
                rule_name="Maximum Image Size",
                dataset=dataset,
                image_path=str(image_path),
                message=(
                    f"Image resolution ({width} × {height}) "
                    f"exceeds the supported maximum "
                    f"({max_width} × {max_height})."
                ),
                recommendation="Resize the image before training.",
            )
        )
# =============================================================================
# Duplicate Image Rules
# =============================================================================

def check_duplicate_image_path(
    image_path: str | Path,
    dataset: str,
    seen_paths: set[Path],
    result: ValidationResult,
) -> None:
    """
    Detect duplicate image paths.
    """

    image_path = normalize_path(image_path)

    if image_path in seen_paths:

        result.add_issue(
            ValidationIssue(
                issue_id="IMG007",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DUPLICATE,
                rule_name="Duplicate Image Path",
                dataset=dataset,
                image_path=str(image_path),
                message="Duplicate image path detected.",
                recommendation="Remove duplicate image entries.",
            )
        )

    else:

        seen_paths.add(image_path)


# =============================================================================
# Image Metadata Rules
# =============================================================================

def check_image_metadata(
    image_path: str | Path,
    dataset: str,
    result: ValidationResult,
) -> None:
    """
    Validate image metadata.
    """

    image_path = normalize_path(image_path)

    if not path_exists(image_path):
        return

    width, height = get_image_size(image_path)

    if width == 0 or height == 0:

        result.add_issue(
            ValidationIssue(
                issue_id="IMG008",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.METADATA,
                rule_name="Image Metadata",
                dataset=dataset,
                image_path=str(image_path),
                message="Unable to read image metadata.",
                recommendation="Verify image integrity.",
            )
        )


# =============================================================================
# Image Statistics
# =============================================================================

def update_image_statistics(
    image_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Update image statistics stored in ValidationResult.
    """

    image_path = normalize_path(image_path)

    width, height = get_image_size(image_path)

    if width == 0 or height == 0:
        return

    stats = result.statistics

    stats.total_images += 1

    stats.average_width += width
    stats.average_height += height

    if stats.minimum_width == 0:
        stats.minimum_width = width
    else:
        stats.minimum_width = min(
            stats.minimum_width,
            width,
        )

    if stats.maximum_width == 0:
        stats.maximum_width = width
    else:
        stats.maximum_width = max(
            stats.maximum_width,
            width,
        )

    if stats.minimum_height == 0:
        stats.minimum_height = height
    else:
        stats.minimum_height = min(
            stats.minimum_height,
            height,
        )

    if stats.maximum_height == 0:
        stats.maximum_height = height
    else:
        stats.maximum_height = max(
            stats.maximum_height,
            height,
        )

    image_extension = get_extension(image_path)

    stats.image_formats.setdefault(
        image_extension,
        0,
    )

    stats.image_formats[image_extension] += 1
# =============================================================================
# Image Validation Pipeline
# =============================================================================

def validate_image(
    image_path: str | Path,
    dataset: str,
    result: ValidationResult,
    seen_paths: set[Path],
) -> ValidationResult:
    """
    Execute all image-level validation rules.
    """

    check_image_exists(
        image_path,
        dataset,
        result,
    )

    # Stop if image does not exist
    if result.error_count > 0:
        return result

    check_image_extension(
        image_path,
        dataset,
        result,
    )

    check_image_corruption(
        image_path,
        dataset,
        result,
    )

    check_empty_image(
        image_path,
        dataset,
        result,
    )

    check_image_dimensions(
        image_path,
        dataset,
        result,
    )

    check_duplicate_image_path(
        image_path,
        dataset,
        seen_paths,
        result,
    )

    check_image_metadata(
        image_path,
        dataset,
        result,
    )

    update_image_statistics(
        image_path,
        result,
    )

    # ---------------------------------------------------------
    # Finalize averages
    # ---------------------------------------------------------

    stats = result.statistics

    if stats.total_images > 0:

        stats.average_width = (
            stats.average_width /
            stats.total_images
        )

        stats.average_height = (
            stats.average_height /
            stats.total_images
        )

    return result


# =============================================================================
# Public API
# =============================================================================

__all__ = [

    "check_image_exists",

    "check_image_extension",

    "check_image_corruption",

    "check_empty_image",

    "check_image_dimensions",

    "check_duplicate_image_path",

    "check_image_metadata",

    "update_image_statistics",

    "validate_image",
]