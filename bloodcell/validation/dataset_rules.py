"""
BloodCellAI Framework
Validation Engine

File:
    dataset_rules.py

Description
-----------
Dataset-level validation rules.

These rules validate the overall dataset before image and
annotation validation begins.

Version:
    1.0.0
"""

from __future__ import annotations

import logging

from .models import ValidationCategory

logger = logging.getLogger(__name__)


def validate_dataset(validator, dataset) -> None:
    """
    Execute all dataset-level validation rules.

    Parameters
    ----------
    validator : DatasetValidator

    dataset : UniversalDataset
    """

    logger.info("Running dataset validation rules...")

    check_dataset_exists(validator, dataset)
    check_dataset_name(validator, dataset)
    check_image_collection(validator, dataset)
    check_class_information(validator, dataset)

    logger.info("Dataset validation rules completed.")


###############################################################################
# Rule 1
###############################################################################

def check_dataset_exists(validator, dataset):

    if dataset is None:

        validator._add_error(
            ValidationCategory.DATASET,
            "Dataset object is None.",
            "Pass a valid UniversalDataset."
        )


###############################################################################
# Rule 2
###############################################################################

def check_dataset_name(validator, dataset):

    if dataset is None:
        return

    # UniversalDataset does not carry a single top-level "name" field
    # (it can hold images merged from several source datasets at once).
    # Use dataset_counts -- populated by UniversalDataset.add() -- as the
    # signal for "at least one source dataset is identified".
    dataset_counts = getattr(dataset, "dataset_counts", None)

    if not dataset_counts:

        validator._add_warning(
            ValidationCategory.DATASET,
            "No source dataset names recorded (dataset_counts is empty).",
            "Ensure images are added via UniversalDataset.add() so "
            "dataset_counts is populated."
        )


###############################################################################
# Rule 3
###############################################################################

def check_image_collection(validator, dataset):

    if dataset is None:
        return

    images = getattr(dataset, "images", None)

    if images is None:

        validator._add_error(
            ValidationCategory.DATASET,
            "Dataset has no image collection.",
            "Initialize dataset.images."
        )

        return

    if len(images) == 0:

        validator._add_warning(
            ValidationCategory.DATASET,
            "Dataset contains zero images.",
            "Add image objects before validation."
        )


###############################################################################
# Rule 4
###############################################################################

def check_class_information(validator, dataset):

    if dataset is None:
        return

    # UniversalDataset tracks class distribution in class_counts
    # (populated by UniversalDataset.add()), not a separate "classes" field.
    class_counts = getattr(dataset, "class_counts", None)

    if class_counts is None:

        validator._add_warning(
            ValidationCategory.DATASET,
            "Class information is unavailable (class_counts is None).",
            "Ensure UniversalDataset.class_counts is initialized."
        )

        return

    if len(class_counts) == 0:

        validator._add_warning(
            ValidationCategory.DATASET,
            "No classes found (class_counts is empty).",
            "Verify that annotations were parsed and at least one "
            "object was added to each image."
        )