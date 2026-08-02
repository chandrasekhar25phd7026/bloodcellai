"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    image_rules.py

Version:
    2.0.0

Status:
    Stable

Purpose:
    Image-level validation rules for UniversalDataset.

Description
-----------
This module validates every image contained in a UniversalDataset.

Implemented Rules
-----------------
1. Image path exists
2. Duplicate filename detection
3. Supported image extension
4. Image dimensions
5. Missing image dimensions
6. Dataset consistency
7. Empty annotation list
8. Duplicate image paths

Author:
    Sekhar Muthangi

Project:
    BloodCellAI
===============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Set

from .models import ValidationCategory

from .utils import (
    file_exists,
    get_extension,
    is_supported_image,
)

logger = logging.getLogger(__name__)


###############################################################################
# Supported Formats
###############################################################################

SUPPORTED_IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


###############################################################################
# Rule 1
# Image Path Validation
###############################################################################

def check_image_path(
    validator,
    image,
) -> None:
    """
    Validate image path.
    """

    path = getattr(image, "image_path", None)

    if not path:

        validator._add_error(
            ValidationCategory.IMAGE,
            "Image path is missing.",
            "Assign image.image_path."
        )

        return

    if not file_exists(path):

        validator._add_error(
            ValidationCategory.IMAGE,
            f"Image file not found: {path}",
            "Verify the image location.",
            image_path=path,
        )


###############################################################################
# Rule 2
# Duplicate Filename Validation
###############################################################################

def check_duplicate_filename(
    validator,
    image,
    seen_filenames: Set[str],
) -> None:
    """
    Detect duplicate image files.

    Compares full paths, not just basenames. Basename-only comparison
    used to flag every folder-per-class classification dataset (e.g.
    LISC: Neutrophil/img_0.bmp, Lymphocyte/img_0.bmp, ...) as having
    "duplicate filenames" on every single image, which is a completely
    normal, ubiquitous convention for classification datasets and not
    a real data-quality issue at all -- confirmed while validating the
    new folder-per-class classification path.
    """

    path = getattr(image, "image_path", "")

    if not path:
        return

    normalized_path = str(Path(path)).lower()

    if normalized_path in seen_filenames:

        validator._add_warning(
            ValidationCategory.IMAGE,
            f"Duplicate image file detected: {path}",
            "Remove or rename the duplicate image file.",
            image_path=path,
        )

    else:

        seen_filenames.add(normalized_path)


###############################################################################
# Rule 3
# Image Extension Validation
###############################################################################

def check_image_extension(
    validator,
    image,
) -> None:
    """
    Validate supported image format.
    """

    path = getattr(image, "image_path", "")

    if not path:
        return

    extension = get_extension(path)

    if not is_supported_image(path):

        validator._add_warning(
            ValidationCategory.IMAGE,
            f"Unsupported image format: {extension}",
            "Convert image to PNG, JPG, BMP or TIFF.",
            image_path=path,
        )


###############################################################################
# Rule 4
# Image Size Validation
###############################################################################

def check_image_size(
    validator,
    image,
) -> None:
    """
    Validate image width and height.
    """

    width = getattr(image, "width", None)
    height = getattr(image, "height", None)

    if width is None or height is None:

        validator._add_warning(
            ValidationCategory.IMAGE,
            "Image dimensions are unavailable.",
            "Populate image.width and image.height."
        )

        return

    if width <= 0 or height <= 0:

        validator._add_error(
            ValidationCategory.IMAGE,
            f"Invalid image dimensions ({width} × {height}).",
            "Width and height must be greater than zero."
        )
###############################################################################
# Rule 5
# Missing Image Metadata
###############################################################################

def check_missing_metadata(
    validator,
    image,
) -> None:
    """
    Validate mandatory image metadata.
    """

    required_fields = {
        "image_id": getattr(image, "image_id", None),
        "dataset": getattr(image, "dataset", None),
    }

    image_path = getattr(image, "image_path", None)

    for field_name, value in required_fields.items():

        if value is None or value == "":

            validator._add_warning(
                ValidationCategory.IMAGE,
                f"Missing image metadata: {field_name}.",
                f"Populate image.{field_name}.",
                image_path=image_path,
            )


###############################################################################
# Rule 6
# Dataset Membership Validation
###############################################################################

def check_dataset_consistency(
    validator,
    image,
) -> None:
    """
    Validate dataset membership.

    Every image should specify the dataset from which it
    originated (e.g., BCCD, Raabin-WBC, ALL-IDB).

    UniversalDataset is designed to store images from multiple
    datasets, so we only verify that the dataset field exists.
    """

    dataset_name = getattr(image, "dataset", None)

    if dataset_name is None:

        validator._add_warning(
            ValidationCategory.IMAGE,
            "Image dataset is missing.",
            "Assign image.dataset.",
            image_path=getattr(image, "image_path", None),
        )

        return

    if isinstance(dataset_name, str):

        dataset_name = dataset_name.strip()

    if dataset_name == "":

        validator._add_warning(
            ValidationCategory.IMAGE,
            "Image dataset is empty.",
            "Provide a valid dataset name.",
            image_path=getattr(image, "image_path", None),
        )

###############################################################################
# Rule 7
# Empty Annotation List
###############################################################################

def check_empty_object_list(
    validator,
    image,
) -> None:
    """
    Ensure an image contains annotation objects.
    """

    objects = getattr(image, "objects", None)

    if objects is None:

        validator._add_warning(
            ValidationCategory.IMAGE,
            "Image object list is missing.",
            "Initialize image.objects.",
            image_path=getattr(image, "image_path", None),
        )

        return

    if len(objects) == 0:

        validator._add_warning(
            ValidationCategory.IMAGE,
            "Image contains no annotated objects.",
            "Add at least one annotation.",
            image_path=getattr(image, "image_path", None),
        )


###############################################################################
# Rule 8
# Duplicate Image Paths
###############################################################################

def check_duplicate_image_path(
    validator,
    image,
    seen_paths: Set[str],
) -> None:
    """
    Detect duplicate image paths.
    """

    path = getattr(image, "image_path", None)

    if not path:
        return

    normalized_path = str(Path(path).resolve()).lower()

    if normalized_path in seen_paths:

        validator._add_warning(
            ValidationCategory.IMAGE,
            f"Duplicate image path detected: {path}",
            "Remove duplicate image references.",
            image_path=path,
        )

    else:

        seen_paths.add(normalized_path)


###############################################################################
# Helper
###############################################################################

def validate_single_image(
    validator,
    image,
    seen_filenames,
    seen_paths,
):

    check_image_path(
        validator,
        image,
    )

    check_duplicate_filename(
        validator,
        image,
        seen_filenames,
    )

    check_image_extension(
        validator,
        image,
    )

    check_image_size(
        validator,
        image,
    )

    check_missing_metadata(
        validator,
        image,
    )

    check_dataset_consistency(
        validator,
        image,
    )

    check_empty_object_list(
        validator,
        image,
    )

    check_duplicate_image_path(
        validator,
        image,
        seen_paths,
    )


###############################################################################
# Main Validation Entry
###############################################################################
def validate_images(validator, dataset) -> None:
    """
    Execute all image validation rules.
    """

    if dataset is None:
        return

    images = getattr(dataset, "images", None)

    if not images:
        return

    logger.info("Running image validation rules...")

    seen_filenames: Set[str] = set()
    seen_paths: Set[str] = set()

    for image in images:

        validate_single_image(
            validator,
            image,
            seen_filenames,
            seen_paths,
        )

    logger.info("Image validation completed.")
###############################################################################