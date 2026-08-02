"""
BloodCellAI Framework
Validation Engine

File:
    bbox_rules.py

Description
-----------
Bounding box validation rules.

These rules validate the geometric correctness of every annotation.

Version:
    1.0.0
"""

from __future__ import annotations

import logging

from .models import ValidationCategory

logger = logging.getLogger(__name__)


def validate_bounding_boxes(validator, dataset):
    """
    Validate every bounding box in the dataset.
    """

    if dataset is None:
        return

    images = getattr(dataset, "images", None)

    if not images:
        return

    logger.info("Running bounding box validation rules...")

    for image in images:

        objects = getattr(image, "objects", [])

        for index, obj in enumerate(objects):

            check_coordinates(
                validator,
                obj,
                image,
                index,
            )

            check_box_size(
                validator,
                obj,
                image,
                index,
            )

            check_confidence(
                validator,
                obj,
                image,
                index,
            )

    logger.info("Bounding box validation completed.")


###############################################################################
# Rule 1
###############################################################################

def check_coordinates(
    validator,
    obj,
    image,
    index,
):

    values = {
        "xc": getattr(obj, "xc", None),
        "yc": getattr(obj, "yc", None),
        "w": getattr(obj, "w", None),
        "h": getattr(obj, "h", None),
    }

    for name, value in values.items():

        if value is None:

            validator._add_error(
                ValidationCategory.BOUNDING_BOX,
                f"{name} is missing.",
                f"Assign {name}.",
                image_path=image.image_path,
                object_index=index,
            )

            continue

        if value < 0 or value > 1:

            validator._add_error(
                ValidationCategory.BOUNDING_BOX,
                f"{name}={value} is outside YOLO range [0,1].",
                "Normalize bounding box coordinates.",
                image_path=image.image_path,
                object_index=index,
            )


###############################################################################
# Rule 2
###############################################################################

def check_box_size(
    validator,
    obj,
    image,
    index,
):

    width = getattr(obj, "w", 0)
    height = getattr(obj, "h", 0)

    if width <= 0:

        validator._add_error(
            ValidationCategory.BOUNDING_BOX,
            "Bounding box width must be greater than zero.",
            image_path=image.image_path,
            object_index=index,
        )

    if height <= 0:

        validator._add_error(
            ValidationCategory.BOUNDING_BOX,
            "Bounding box height must be greater than zero.",
            image_path=image.image_path,
            object_index=index,
        )


###############################################################################
# Rule 3
###############################################################################

def check_confidence(
    validator,
    obj,
    image,
    index,
):

    confidence = getattr(obj, "confidence", None)

    if confidence is None:
        return

    if confidence < 0 or confidence > 1:

        validator._add_warning(
            ValidationCategory.BOUNDING_BOX,
            f"Confidence {confidence} outside expected range [0,1].",
            "Confidence should be between 0 and 1.",
            image_path=image.image_path,
            object_index=index,
        )