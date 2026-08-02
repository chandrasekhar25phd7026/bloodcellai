"""
BloodCellAI Framework
Validation Engine

File:
    object_rules.py

Description
-----------
Object-level validation rules.

Each annotated object (blood cell) is validated independently.

Version:
    1.0.0
"""

from __future__ import annotations

import logging

from .models import ValidationCategory

logger = logging.getLogger(__name__)


def validate_objects(validator, dataset):
    """
    Validate every annotated object in the dataset.
    """

    if dataset is None:
        return

    images = getattr(dataset, "images", None)

    if not images:
        return

    logger.info("Running object validation rules...")

    object_ids = set()

    for image in images:

        objects = getattr(image, "objects", None)

        if objects is None:
            continue

        for index, obj in enumerate(objects):

            check_class_name(
                validator,
                obj,
                image,
                index,
            )

            check_class_id(
                validator,
                obj,
                image,
                index,
            )

            check_duplicate_object_id(
                validator,
                obj,
                image,
                index,
                object_ids,
            )

            check_bbox_exists(
                validator,
                obj,
                image,
                index,
            )

    logger.info("Object validation completed.")


###############################################################################
# Rule 1
###############################################################################

def check_class_name(
    validator,
    obj,
    image,
    index,
):

    class_name = getattr(obj, "class_name", None)

    if not class_name:

        validator._add_warning(
            ValidationCategory.OBJECT,
            "Missing class name.",
            "Assign object.class_name.",
            image_path=getattr(image, "image_path", None),
            object_index=index,
        )


###############################################################################
# Rule 2
###############################################################################

def check_class_id(
    validator,
    obj,
    image,
    index,
):

    class_id = getattr(obj, "class_id", None)

    if class_id is None:

        validator._add_error(
            ValidationCategory.OBJECT,
            "Missing class id.",
            "Assign object.class_id.",
            image_path=getattr(image, "image_path", None),
            object_index=index,
        )

        return

    if class_id < 0:

        validator._add_error(
            ValidationCategory.OBJECT,
            "Negative class id.",
            "Class id must be >= 0.",
            image_path=getattr(image, "image_path", None),
            object_index=index,
        )


###############################################################################
# Rule 3
###############################################################################

def check_duplicate_object_id(
    validator,
    obj,
    image,
    index,
    object_ids,
):

    object_id = getattr(obj, "object_id", None)

    if object_id is None:
        return

    if object_id in object_ids:

        validator._add_warning(
            ValidationCategory.OBJECT,
            f"Duplicate object id : {object_id}",
            "Assign a unique object id.",
            image_path=getattr(image, "image_path", None),
            object_index=index,
        )

    else:

        object_ids.add(object_id)


###############################################################################
# Rule 4
###############################################################################

def check_bbox_exists(
    validator,
    obj,
    image,
    index,
):
    """
    Confirm the object carries bounding-box coordinates at all.

    This is a presence check only (are xc/yc/w/h set?); geometric
    correctness (in-range, non-degenerate, etc.) is handled separately
    by bbox_rules.py.
    """

    required_fields = ("xc", "yc", "w", "h")

    missing = [
        field_name
        for field_name in required_fields
        if getattr(obj, field_name, None) is None
    ]

    if missing:

        validator._add_error(
            ValidationCategory.BOUNDING_BOX,
            f"Object is missing bounding-box field(s): {', '.join(missing)}.",
            "Ensure the adapter populates xc, yc, w, and h.",
            image_path=getattr(image, "image_path", None),
            object_index=index,
        )

