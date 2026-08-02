"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    dataset_rules.py

Version:
    2.0.0

Purpose:
    Dataset-level validation rules.

Description:
    This module validates the overall dataset structure before
    image-level and object-level validation begins.

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
    count_annotations,
    count_images,
    get_annotation_files,
    get_image_files,
    is_directory,
    is_duplicate_filename,
    is_empty,
    normalize_path,
    path_exists,
)

# =============================================================================
# Dataset Rules
# =============================================================================


def check_dataset_exists(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Check whether the dataset directory exists.
    """

    dataset_path = normalize_path(dataset_path)

    if not path_exists(dataset_path):

        result.add_issue(
            ValidationIssue(
                issue_id="DS001",
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.DATASET,
                rule_name="Dataset Exists",
                dataset=dataset_path.name,
                message="Dataset directory does not exist.",
                recommendation="Provide a valid dataset directory.",
            )
        )


def check_dataset_directory(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Ensure the dataset path is a directory.
    """

    dataset_path = normalize_path(dataset_path)

    if not is_directory(dataset_path):

        result.add_issue(
            ValidationIssue(
                issue_id="DS002",
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.DATASET,
                rule_name="Dataset Directory",
                dataset=dataset_path.name,
                message="Dataset path is not a directory.",
                recommendation="Select a valid dataset folder.",
            )
        )
# =============================================================================
# Dataset Content Rules
# =============================================================================

def check_images_exist(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Ensure the dataset contains image files.
    """

    dataset_path = normalize_path(dataset_path)

    image_count = count_images(dataset_path)

    if image_count == 0:

        result.add_issue(
            ValidationIssue(
                issue_id="DS003",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATASET,
                rule_name="Images Exist",
                dataset=dataset_path.name,
                message="No image files were found.",
                recommendation="Add supported image files to the dataset.",
            )
        )


def check_annotations_exist(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Ensure the dataset contains annotation files.
    """

    dataset_path = normalize_path(dataset_path)

    annotation_count = count_annotations(dataset_path)

    if annotation_count == 0:

        result.add_issue(
            ValidationIssue(
                issue_id="DS004",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATASET,
                rule_name="Annotations Exist",
                dataset=dataset_path.name,
                message="No annotation files were found.",
                recommendation="Add annotation files to the dataset.",
            )
        )


def check_dataset_empty(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Check whether the dataset directory is empty.
    """

    dataset_path = normalize_path(dataset_path)

    files = list(dataset_path.iterdir())

    if len(files) == 0:

        result.add_issue(
            ValidationIssue(
                issue_id="DS005",
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.DATASET,
                rule_name="Dataset Empty",
                dataset=dataset_path.name,
                message="Dataset directory is empty.",
                recommendation="Populate the dataset before validation.",
            )
        )


def check_image_annotation_count(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Compare the number of images and annotations.
    """

    dataset_path = normalize_path(dataset_path)

    image_count = count_images(dataset_path)
    annotation_count = count_annotations(dataset_path)

    if image_count != annotation_count:

        result.add_issue(
            ValidationIssue(
                issue_id="DS006",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DATASET,
                rule_name="Image Annotation Count",
                dataset=dataset_path.name,
                message=(
                    f"Found {image_count} images but "
                    f"{annotation_count} annotations."
                ),
                recommendation=(
                    "Verify that every image has a matching annotation."
                ),
            )
        )
# =============================================================================
# Dataset Consistency Rules
# =============================================================================

def check_missing_annotations(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Check whether every image has a corresponding annotation file.
    """

    dataset_path = normalize_path(dataset_path)

    image_files = get_image_files(dataset_path)
    annotation_files = get_annotation_files(dataset_path)

    annotation_names = {
        annotation.stem
        for annotation in annotation_files
    }

    for image in image_files:

        if image.stem not in annotation_names:

            result.add_issue(
                ValidationIssue(
                    issue_id="DS007",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATASET,
                    rule_name="Missing Annotation",
                    dataset=dataset_path.name,
                    image_path=str(image),
                    message=f"Missing annotation for '{image.name}'.",
                    recommendation="Create the corresponding annotation file.",
                )
            )


def check_duplicate_filenames(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Detect duplicate image filenames.
    """

    dataset_path = normalize_path(dataset_path)

    seen_filenames = set()

    for image in get_image_files(dataset_path):

        if is_duplicate_filename(
            image.name,
            seen_filenames,
        ):

            result.add_issue(
                ValidationIssue(
                    issue_id="DS008",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DUPLICATE,
                    rule_name="Duplicate Filename",
                    dataset=dataset_path.name,
                    image_path=str(image),
                    message=f"Duplicate filename '{image.name}'.",
                    recommendation="Rename duplicate image files.",
                )
            )


def check_empty_annotation_files(
    dataset_path: str | Path,
    result: ValidationResult,
) -> None:
    """
    Detect empty annotation files.
    """

    dataset_path = normalize_path(dataset_path)

    for annotation in get_annotation_files(dataset_path):

        try:

            if annotation.stat().st_size == 0:

                result.add_issue(
                    ValidationIssue(
                        issue_id="DS009",
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.DATASET,
                        rule_name="Empty Annotation",
                        dataset=dataset_path.name,
                        image_path=str(annotation),
                        message=f"Annotation '{annotation.name}' is empty.",
                        recommendation="Populate the annotation file.",
                    )
                )

        except OSError:

            result.add_issue(
                ValidationIssue(
                    issue_id="DS010",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATASET,
                    rule_name="Unreadable Annotation",
                    dataset=dataset_path.name,
                    image_path=str(annotation),
                    message=f"Cannot read '{annotation.name}'.",
                    recommendation="Check file permissions or file integrity.",
                )
            )
# =============================================================================
# Dataset Metadata Rules
# =============================================================================

def check_dataset_name(
    dataset_name: str,
    result: ValidationResult,
) -> None:
    """
    Validate dataset name.
    """

    if is_empty(dataset_name):

        result.add_issue(
            ValidationIssue(
                issue_id="DS011",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.METADATA,
                rule_name="Dataset Name",
                dataset="Unknown",
                message="Dataset name is empty.",
                recommendation="Provide a dataset name.",
            )
        )


# =============================================================================
# Dataset Validation Pipeline
# =============================================================================

def validate_dataset(
    dataset_path: str | Path,
    result: ValidationResult,
) -> ValidationResult:
    """
    Execute all dataset-level validation rules.
    """

    check_dataset_exists(
        dataset_path,
        result,
    )

    check_dataset_directory(
        dataset_path,
        result,
    )

    # Stop if dataset does not exist
    if result.critical_count > 0:
        return result

    check_dataset_empty(
        dataset_path,
        result,
    )

    check_images_exist(
        dataset_path,
        result,
    )

    check_annotations_exist(
        dataset_path,
        result,
    )

    check_image_annotation_count(
        dataset_path,
        result,
    )

    check_missing_annotations(
        dataset_path,
        result,
    )

    check_duplicate_filenames(
        dataset_path,
        result,
    )

    check_empty_annotation_files(
        dataset_path,
        result,
    )

    return result


# =============================================================================
# Public API
# =============================================================================

__all__ = [

    "check_dataset_exists",
    "check_dataset_directory",
    "check_dataset_empty",

    "check_images_exist",
    "check_annotations_exist",

    "check_image_annotation_count",

    "check_missing_annotations",

    "check_duplicate_filenames",

    "check_empty_annotation_files",

    "check_dataset_name",

    "validate_dataset",
]