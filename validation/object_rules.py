"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    object_rules.py

Version:
    2.0.0

Purpose:
    Object-level validation rules.

Description:
    Validates parsed annotation objects independently of the
    original annotation format (YOLO, Pascal VOC, COCO, etc.).

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from .models import (
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

# =============================================================================
# Object Existence Rules
# =============================================================================


def check_empty_object_list(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Ensure that at least one object exists.
    """

    if len(objects) == 0:

        result.add_issue(
            ValidationIssue(
                issue_id="OBJ001",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.OBJECT,
                rule_name="Empty Object List",
                dataset=dataset,
                image_path=image_path,
                message="No annotated objects were found.",
                recommendation="Verify the annotation file.",
            )
        )


# =============================================================================
# Class Validation Rules
# =============================================================================


def check_missing_class_id(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Ensure every object contains a class_id.
    """

    for index, obj in enumerate(objects):

        if "class_id" not in obj:

            result.add_issue(
                ValidationIssue(
                    issue_id="OBJ002",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.OBJECT,
                    rule_name="Missing Class ID",
                    dataset=dataset,
                    image_path=image_path,
                    message=f"Object {index} has no class_id.",
                    recommendation="Assign a valid class ID.",
                )
            )


def check_missing_class_name(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Ensure every object contains a class_name.
    """

    for index, obj in enumerate(objects):

        if "class_name" not in obj:

            result.add_issue(
                ValidationIssue(
                    issue_id="OBJ003",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.OBJECT,
                    rule_name="Missing Class Name",
                    dataset=dataset,
                    image_path=image_path,
                    message=f"Object {index} has no class_name.",
                    recommendation="Provide the class name if available.",
                )
            )
# =============================================================================
# Class Validation Rules
# =============================================================================

def check_invalid_class_id(
    objects: list[dict],
    valid_class_ids: set[int],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Verify that every object has a valid class ID.
    """

    for index, obj in enumerate(objects):

        if "class_id" not in obj:
            continue

        class_id = obj["class_id"]

        if class_id not in valid_class_ids:

            result.add_issue(
                ValidationIssue(
                    issue_id="OBJ004",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.OBJECT,
                    rule_name="Invalid Class ID",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Object {index} has invalid "
                        f"class_id '{class_id}'."
                    ),
                    recommendation="Use one of the supported class IDs.",
                )
            )


# =============================================================================
# Duplicate Object Rules
# =============================================================================

def check_duplicate_objects(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Detect duplicate annotation objects.
    """

    seen = set()

    for index, obj in enumerate(objects):

        class_id = obj.get("class_id", -1)
        bbox = tuple(obj.get("bbox", []))

        key = (
            class_id,
            bbox,
        )

        if key in seen:

            result.add_issue(
                ValidationIssue(
                    issue_id="OBJ005",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DUPLICATE,
                    rule_name="Duplicate Object",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Duplicate object detected "
                        f"at index {index}."
                    ),
                    recommendation="Remove duplicate annotations.",
                )
            )

        else:
            seen.add(key)


# =============================================================================
# Confidence Validation Rules
# =============================================================================

def check_invalid_confidence(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Validate prediction confidence values.

    Ground-truth annotations usually do not contain confidence,
    therefore confidence=None is considered valid.
    """

    for index, obj in enumerate(objects):

        confidence = obj.get("confidence", None)

        if confidence is None:
            continue

        if not isinstance(confidence, (int, float)):

            result.add_issue(
                ValidationIssue(
                    issue_id="OBJ006",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.OBJECT,
                    rule_name="Confidence Type",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Object {index} has a non-numeric "
                        f"confidence value."
                    ),
                    recommendation="Confidence must be numeric.",
                )
            )
            continue

        if confidence < 0.0 or confidence > 1.0:

            result.add_issue(
                ValidationIssue(
                    issue_id="OBJ007",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.OBJECT,
                    rule_name="Confidence Range",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Object {index} has confidence "
                        f"{confidence:.3f} outside [0, 1]."
                    ),
                    recommendation="Confidence must be between 0 and 1.",
                )
            )
# =============================================================================
# Object Statistics
# =============================================================================

def update_object_statistics(
    objects: list[dict],
    dataset: str,
    result: ValidationResult,
) -> None:
    """
    Update object statistics stored in ValidationResult.
    """

    stats = result.statistics

    # ---------------------------------------------------------
    # Total objects
    # ---------------------------------------------------------

    stats.total_objects += len(objects)

    # ---------------------------------------------------------
    # Dataset counts
    # ---------------------------------------------------------

    stats.dataset_counts.setdefault(dataset, 0)
    stats.dataset_counts[dataset] += len(objects)

    # ---------------------------------------------------------
    # Class distribution
    # ---------------------------------------------------------

    for obj in objects:

        class_name = obj.get("class_name")

        if class_name is None:
            class_name = str(obj.get("class_id", "Unknown"))

        stats.class_counts.setdefault(class_name, 0)
        stats.class_counts[class_name] += 1

    stats.number_of_classes = len(stats.class_counts)


# =============================================================================
# Class Distribution Rules
# =============================================================================

def check_missing_classes(
    objects: list[dict],
    expected_classes: set[str],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Check whether expected classes are present.
    """

    if not expected_classes:
        return

    present_classes = {
        obj.get("class_name")
        for obj in objects
        if obj.get("class_name") is not None
    }

    missing = expected_classes - present_classes

    if missing:

        result.add_issue(
            ValidationIssue(
                issue_id="OBJ008",
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.OBJECT,
                rule_name="Missing Classes",
                dataset=dataset,
                image_path=image_path,
                message=(
                    "Missing expected classes: "
                    + ", ".join(sorted(missing))
                ),
                recommendation="Verify whether all expected classes are represented.",
            )
        )


def check_single_class_image(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Detect images containing only one class.
    """

    class_names = {
        obj.get("class_name")
        for obj in objects
        if obj.get("class_name") is not None
    }

    if len(class_names) == 1 and len(objects) > 1:

        result.add_issue(
            ValidationIssue(
                issue_id="OBJ009",
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.OBJECT,
                rule_name="Single Class Image",
                dataset=dataset,
                image_path=image_path,
                message=(
                    "Image contains objects from only one class."
                ),
                recommendation=(
                    "This may be expected depending on the dataset."
                ),
            )
        )
# =============================================================================
# Object Validation Pipeline
# =============================================================================

def validate_objects(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
    valid_class_ids: set[int] | None = None,
    expected_classes: set[str] | None = None,
) -> ValidationResult:
    """
    Execute all object validation rules.
    """

    if valid_class_ids is None:
        valid_class_ids = set()

    if expected_classes is None:
        expected_classes = set()

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    check_empty_object_list(
        objects,
        dataset,
        image_path,
        result,
    )

    if len(objects) == 0:
        return result

    # ---------------------------------------------------------
    # Class validation
    # ---------------------------------------------------------

    check_missing_class_id(
        objects,
        dataset,
        image_path,
        result,
    )

    check_missing_class_name(
        objects,
        dataset,
        image_path,
        result,
    )

    if len(valid_class_ids) > 0:

        check_invalid_class_id(
            objects,
            valid_class_ids,
            dataset,
            image_path,
            result,
        )

    # ---------------------------------------------------------
    # Duplicate validation
    # ---------------------------------------------------------

    check_duplicate_objects(
        objects,
        dataset,
        image_path,
        result,
    )

    # ---------------------------------------------------------
    # Confidence validation
    # ---------------------------------------------------------

    check_invalid_confidence(
        objects,
        dataset,
        image_path,
        result,
    )

    # ---------------------------------------------------------
    # Class distribution
    # ---------------------------------------------------------

    if len(expected_classes) > 0:

        check_missing_classes(
            objects,
            expected_classes,
            dataset,
            image_path,
            result,
        )

    check_single_class_image(
        objects,
        dataset,
        image_path,
        result,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    update_object_statistics(
        objects,
        dataset,
        result,
    )

    return result


# =============================================================================
# Public API
# =============================================================================

__all__ = [

    # Object existence
    "check_empty_object_list",

    # Class validation
    "check_missing_class_id",
    "check_missing_class_name",
    "check_invalid_class_id",

    # Duplicate validation
    "check_duplicate_objects",

    # Confidence validation
    "check_invalid_confidence",

    # Statistics
    "update_object_statistics",

    # Distribution
    "check_missing_classes",
    "check_single_class_image",

    # Pipeline
    "validate_objects",
]