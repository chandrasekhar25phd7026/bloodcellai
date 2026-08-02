"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    bbox_rules.py

Version:
    4.0.0

Purpose:
    Bounding Box Validation Rules

Description:
    This module validates bounding boxes present in object detection datasets.
    It supports both Pascal VOC and YOLO annotation formats.

Supported Formats
-----------------
- Pascal VOC
- YOLO

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from .geometry import (
    bbox_area,
    bbox_height,
    bbox_width,
    calculate_iou,
)

from .models import (
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

# =============================================================================
# Bounding Box Structure Rules
# =============================================================================


def check_missing_bbox(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Verify that every object contains a bounding box.
    """

    for index, obj in enumerate(objects):

        if "bbox" not in obj:

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX001",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Missing Bounding Box",
                    dataset=dataset,
                    image_path=image_path,
                    message=f"Object {index} has no bounding box.",
                    recommendation="Provide a valid bounding box.",
                )
            )


def check_bbox_length(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Ensure every bounding box contains exactly four coordinates.
    """

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if bbox is None:
            continue

        if not isinstance(bbox, (list, tuple)):

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX002",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Invalid Bounding Box Type",
                    dataset=dataset,
                    image_path=image_path,
                    message=f"Object {index} bounding box is not a list or tuple.",
                    recommendation="Store bounding boxes as a list of four values.",
                )
            )

            continue

        if len(bbox) != 4:

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX003",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Invalid Bounding Box Length",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Object {index} contains "
                        f"{len(bbox)} coordinates instead of four."
                    ),
                    recommendation="Bounding boxes must contain exactly four coordinates.",
                )
            )


def check_numeric_bbox(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Verify every bounding-box coordinate is numeric.
    """

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if bbox is None:
            continue

        if not isinstance(bbox, (list, tuple)):
            continue

        if len(bbox) != 4:
            continue

        for coordinate_index, coordinate in enumerate(bbox):

            if not isinstance(coordinate, (int, float)):

                result.add_issue(
                    ValidationIssue(
                        issue_id="BOX004",
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.BOUNDING_BOX,
                        rule_name="Non-numeric Coordinate",
                        dataset=dataset,
                        image_path=image_path,
                        message=(
                            f"Object {index} coordinate "
                            f"{coordinate_index} "
                            f"('{coordinate}') is not numeric."
                        ),
                        recommendation=(
                            "Bounding-box coordinates must be integers or floats."
                        ),
                    )
                )

                break
# =============================================================================
# Bounding Box Geometry Rules
# =============================================================================


def check_negative_coordinates(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
    bbox_format: str = "voc",
) -> None:
    """
    Detect negative coordinates or negative width/height.
    """

    bbox_format = bbox_format.lower()

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if (
            bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        if bbox_format == "yolo":

            _, _, width, height = bbox

            if width < 0 or height < 0:

                result.add_issue(
                    ValidationIssue(
                        issue_id="BOX005",
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.BOUNDING_BOX,
                        rule_name="Negative Box Size",
                        dataset=dataset,
                        image_path=image_path,
                        message=f"Object {index} has negative width or height.",
                        recommendation="Bounding-box width and height must be positive.",
                    )
                )

        else:

            x1, y1, x2, y2 = bbox

            if min(x1, y1, x2, y2) < 0:

                result.add_issue(
                    ValidationIssue(
                        issue_id="BOX006",
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.BOUNDING_BOX,
                        rule_name="Negative Coordinate",
                        dataset=dataset,
                        image_path=image_path,
                        message=f"Object {index} contains negative coordinates.",
                        recommendation="Coordinates cannot be negative.",
                    )
                )


def check_zero_area_bbox(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
    bbox_format: str = "voc",
) -> None:
    """
    Detect bounding boxes having zero or negative area.
    """

    bbox_format = bbox_format.lower()

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if (
            bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        if bbox_format == "yolo":

            _, _, width, height = bbox

            if width <= 0 or height <= 0:

                result.add_issue(
                    ValidationIssue(
                        issue_id="BOX007",
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.BOUNDING_BOX,
                        rule_name="Zero Area Bounding Box",
                        dataset=dataset,
                        image_path=image_path,
                        message=f"Object {index} has zero-area bounding box.",
                        recommendation="Width and height must be greater than zero.",
                    )
                )

        else:

            width = bbox_width(bbox)
            height = bbox_height(bbox)

            if width <= 0 or height <= 0:

                result.add_issue(
                    ValidationIssue(
                        issue_id="BOX008",
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.BOUNDING_BOX,
                        rule_name="Invalid Bounding Box",
                        dataset=dataset,
                        image_path=image_path,
                        message=f"Object {index} has invalid geometry.",
                        recommendation="Ensure x2>x1 and y2>y1.",
                    )
                )


def check_bbox_inside_image(
    objects: list[dict],
    image_width: int,
    image_height: int,
    dataset: str,
    image_path: str,
    result: ValidationResult,
    bbox_format: str = "voc",
) -> None:
    """
    Verify every Pascal VOC bounding box lies completely inside the image.
    """

    if bbox_format.lower() != "voc":
        return

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if (
            bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        x1, y1, x2, y2 = bbox

        if (
            x1 < 0
            or y1 < 0
            or x2 > image_width
            or y2 > image_height
        ):

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX009",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Bounding Box Outside Image",
                    dataset=dataset,
                    image_path=image_path,
                    message=f"Object {index} extends outside image boundaries.",
                    recommendation="Clip or correct the annotation.",
                )
            )


def check_normalized_yolo_bbox(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Verify YOLO coordinates are normalized to the range [0,1].
    """

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if (
            bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        x_center, y_center, width, height = bbox

        if (
            not (0.0 <= x_center <= 1.0)
            or not (0.0 <= y_center <= 1.0)
            or width <= 0
            or width > 1
            or height <= 0
            or height > 1
        ):

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX010",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Invalid YOLO Bounding Box",
                    dataset=dataset,
                    image_path=image_path,
                    message=f"Object {index} contains invalid normalized coordinates.",
                    recommendation="YOLO coordinates must be normalized to the range [0,1].",
                )
            )
# =============================================================================
# Bounding Box Overlap Rules
# =============================================================================


def check_duplicate_bboxes(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
) -> None:
    """
    Detect duplicate bounding boxes.
    """

    total = len(objects)

    for i in range(total):

        bbox1 = objects[i].get("bbox")

        if (
            bbox1 is None
            or not isinstance(bbox1, (list, tuple))
            or len(bbox1) != 4
            or not all(isinstance(v, (int, float)) for v in bbox1)
        ):
            continue

        for j in range(i + 1, total):

            bbox2 = objects[j].get("bbox")

            if (
                bbox2 is None
                or not isinstance(bbox2, (list, tuple))
                or len(bbox2) != 4
                or not all(isinstance(v, (int, float)) for v in bbox2)
            ):
                continue

            if tuple(bbox1) == tuple(bbox2):

                result.add_issue(
                    ValidationIssue(
                        issue_id="BOX011",
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.BOUNDING_BOX,
                        rule_name="Duplicate Bounding Box",
                        dataset=dataset,
                        image_path=image_path,
                        message=(
                            f"Objects {i} and {j} have identical bounding boxes."
                        ),
                        recommendation="Remove duplicate annotations.",
                    )
                )


def check_overlapping_bboxes(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
    iou_threshold: float = 0.85,
) -> None:
    """
    Detect excessive overlap between Pascal VOC bounding boxes.
    """

    total = len(objects)

    for i in range(total):

        bbox1 = objects[i].get("bbox")

        if (
            bbox1 is None
            or not isinstance(bbox1, (list, tuple))
            or len(bbox1) != 4
            or not all(isinstance(v, (int, float)) for v in bbox1)
        ):
            continue

        for j in range(i + 1, total):

            bbox2 = objects[j].get("bbox")

            if (
                bbox2 is None
                or not isinstance(bbox2, (list, tuple))
                or len(bbox2) != 4
                or not all(isinstance(v, (int, float)) for v in bbox2)
            ):
                continue

            iou = calculate_iou(bbox1, bbox2)

            if iou >= iou_threshold:

                result.add_issue(
                    ValidationIssue(
                        issue_id="BOX012",
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.BOUNDING_BOX,
                        rule_name="Highly Overlapping Bounding Boxes",
                        dataset=dataset,
                        image_path=image_path,
                        message=(
                            f"Objects {i} and {j} overlap "
                            f"(IoU = {iou:.3f})."
                        ),
                        recommendation=(
                            "Review the annotations for duplicate or "
                            "incorrect bounding boxes."
                        ),
                    )
                )
# =============================================================================
# Bounding Box Size Rules
# =============================================================================


def check_tiny_bboxes(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
    min_size: float = 3.0,
) -> None:
    """
    Detect extremely small bounding boxes.
    """

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if (
            bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        width = bbox_width(bbox)
        height = bbox_height(bbox)

        if width < min_size or height < min_size:

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX013",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Tiny Bounding Box",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Object {index} has a very small bounding box "
                        f"({width:.1f} × {height:.1f} pixels)."
                    ),
                    recommendation=(
                        "Verify whether the annotation is correct."
                    ),
                )
            )


def check_large_bboxes(
    objects: list[dict],
    image_width: int,
    image_height: int,
    dataset: str,
    image_path: str,
    result: ValidationResult,
    max_ratio: float = 0.95,
) -> None:
    """
    Detect unusually large bounding boxes.
    """

    image_area = image_width * image_height

    if image_area <= 0:
        return

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if (
            bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        area = bbox_area(bbox)

        ratio = area / image_area

        if ratio > max_ratio:

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX014",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Large Bounding Box",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Object {index} occupies "
                        f"{ratio:.1%} of the image."
                    ),
                    recommendation=(
                        "Verify whether the annotation is oversized."
                    ),
                )
            )


def check_aspect_ratio(
    objects: list[dict],
    dataset: str,
    image_path: str,
    result: ValidationResult,
    max_ratio: float = 20.0,
) -> None:
    """
    Detect unrealistic aspect ratios.
    """

    for index, obj in enumerate(objects):

        bbox = obj.get("bbox")

        if (
            bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue

        width = max(1.0, bbox_width(bbox))
        height = max(1.0, bbox_height(bbox))

        ratio = max(width / height, height / width)

        if ratio > max_ratio:

            result.add_issue(
                ValidationIssue(
                    issue_id="BOX015",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.BOUNDING_BOX,
                    rule_name="Extreme Aspect Ratio",
                    dataset=dataset,
                    image_path=image_path,
                    message=(
                        f"Object {index} has an aspect ratio "
                        f"of {ratio:.2f}."
                    ),
                    recommendation=(
                        "Review the annotation geometry."
                    ),
                )
            )
# =============================================================================
# Validation Pipeline
# =============================================================================


def validate_bboxes(
    objects: list[dict],
    image_width: int,
    image_height: int,
    dataset: str,
    image_path: str,
    result: ValidationResult,
    bbox_format: str = "voc",
) -> ValidationResult:
    """
    Execute all bounding-box validation rules.
    """

    bbox_format = bbox_format.lower()

    # -------------------------------------------------------------------------
    # Structure Rules
    # -------------------------------------------------------------------------

    check_missing_bbox(
        objects,
        dataset,
        image_path,
        result,
    )

    check_bbox_length(
        objects,
        dataset,
        image_path,
        result,
    )

    check_numeric_bbox(
        objects,
        dataset,
        image_path,
        result,
    )

    # -------------------------------------------------------------------------
    # Geometry Rules
    # -------------------------------------------------------------------------

    check_negative_coordinates(
        objects,
        dataset,
        image_path,
        result,
        bbox_format,
    )

    check_zero_area_bbox(
        objects,
        dataset,
        image_path,
        result,
        bbox_format,
    )

    if bbox_format == "voc":

        check_bbox_inside_image(
            objects,
            image_width,
            image_height,
            dataset,
            image_path,
            result,
            bbox_format,
        )

    elif bbox_format == "yolo":

        check_normalized_yolo_bbox(
            objects,
            dataset,
            image_path,
            result,
        )

    # -------------------------------------------------------------------------
    # Duplicate / Overlap Rules
    # -------------------------------------------------------------------------

    if bbox_format == "voc":

        check_duplicate_bboxes(
            objects,
            dataset,
            image_path,
            result,
        )

        check_overlapping_bboxes(
            objects,
            dataset,
            image_path,
            result,
        )

    # -------------------------------------------------------------------------
    # Size Rules
    # -------------------------------------------------------------------------

    if bbox_format == "voc":

        check_tiny_bboxes(
            objects,
            dataset,
            image_path,
            result,
        )

        check_large_bboxes(
            objects,
            image_width,
            image_height,
            dataset,
            image_path,
            result,
        )

        check_aspect_ratio(
            objects,
            dataset,
            image_path,
            result,
        )

    return result


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "check_missing_bbox",
    "check_bbox_length",
    "check_numeric_bbox",
    "check_negative_coordinates",
    "check_zero_area_bbox",
    "check_bbox_inside_image",
    "check_normalized_yolo_bbox",
    "check_duplicate_bboxes",
    "check_overlapping_bboxes",
    "check_tiny_bboxes",
    "check_large_bboxes",
    "check_aspect_ratio",
    "validate_bboxes",
]
