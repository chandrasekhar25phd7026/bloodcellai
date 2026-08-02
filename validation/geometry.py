"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    geometry.py

Version:
    3.0.0

Purpose:
    Geometric utility functions for bounding boxes.

Description:
    This module provides reusable geometric functions for object
    detection annotations.

    The functions contained here DO NOT perform validation.
    They only compute geometric properties.

Supported Formats
-----------------
- Pascal VOC
- YOLO

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import math

# =============================================================================
# Internal Helpers
# =============================================================================


def _is_valid_bbox(
    bbox: list[float] | tuple[float, float, float, float],
) -> bool:
    """
    Return True if bbox contains four numeric values.
    """

    if not isinstance(bbox, (list, tuple)):
        return False

    if len(bbox) != 4:
        return False

    return all(isinstance(v, (int, float)) for v in bbox)


# =============================================================================
# Bounding Box Measurements
# =============================================================================


def bbox_width(
    bbox: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Return bounding-box width.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, _, x2, _ = bbox

    return max(0.0, x2 - x1)


def bbox_height(
    bbox: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Return bounding-box height.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    _, y1, _, y2 = bbox

    return max(0.0, y2 - y1)


def bbox_area(
    bbox: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Return bounding-box area.
    """

    return bbox_width(bbox) * bbox_height(bbox)


def bbox_center(
    bbox: list[float] | tuple[float, float, float, float],
) -> tuple[float, float]:
    """
    Return bounding-box center.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )
# =============================================================================
# Bounding Box Format Conversion
# =============================================================================


def voc_to_yolo(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a Pascal VOC bounding box (x1, y1, x2, y2)
    to YOLO format (x_center, y_center, width, height).

    All returned values are normalized to [0, 1].
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def yolo_to_voc(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a YOLO bounding box
    (x_center, y_center, width, height)

    into

    Pascal VOC
    (x1, y1, x2, y2).

    Input values must be normalized.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    x_center, y_center, width, height = bbox

    x_center *= image_width
    y_center *= image_height

    width *= image_width
    height *= image_height

    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0

    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    return (
        x1,
        y1,
        x2,
        y2,
    )


def convert_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    source_format: str,
    target_format: str,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Generic bounding-box conversion.

    Supported formats
    -----------------
    voc
    yolo
    """

    source_format = source_format.lower()
    target_format = target_format.lower()

    if source_format == target_format:
        return tuple(bbox)

    if source_format == "voc" and target_format == "yolo":
        return voc_to_yolo(
            bbox,
            image_width,
            image_height,
        )

    if source_format == "yolo" and target_format == "voc":
        return yolo_to_voc(
            bbox,
            image_width,
            image_height,
        )

    raise ValueError(
        f"Unsupported conversion: "
        f"{source_format} -> {target_format}"
    )
# =============================================================================
# Geometry Operations
# =============================================================================


def intersection_area(
    bbox1: list[float] | tuple[float, float, float, float],
    bbox2: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Compute the intersection area between two Pascal VOC
    bounding boxes.
    """

    if not _is_valid_bbox(bbox1):
        raise ValueError("Invalid first bounding box.")

    if not _is_valid_bbox(bbox2):
        raise ValueError("Invalid second bounding box.")

    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def union_area(
    bbox1: list[float] | tuple[float, float, float, float],
    bbox2: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Compute the union area between two Pascal VOC
    bounding boxes.
    """

    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)

    intersection = intersection_area(
        bbox1,
        bbox2,
    )

    return area1 + area2 - intersection


def calculate_iou(
    bbox1: list[float] | tuple[float, float, float, float],
    bbox2: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Compute Intersection over Union (IoU).

    Returns
    -------
    float
        IoU in the range [0, 1].
    """

    union = union_area(
        bbox1,
        bbox2,
    )

    if union <= 0:
        return 0.0

    intersection = intersection_area(
        bbox1,
        bbox2,
    )

    return intersection / union


def overlap_ratio(
    bbox1: list[float] | tuple[float, float, float, float],
    bbox2: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Compute the overlap ratio relative to the
    smaller bounding box.

    Useful for duplicate annotation detection.
    """

    intersection = intersection_area(
        bbox1,
        bbox2,
    )

    smallest_area = min(
        bbox_area(bbox1),
        bbox_area(bbox2),
    )

    if smallest_area <= 0:
        return 0.0

    return intersection / smallest_area


def boxes_intersect(
    bbox1: list[float] | tuple[float, float, float, float],
    bbox2: list[float] | tuple[float, float, float, float],
) -> bool:
    """
    Return True if two bounding boxes intersect.
    """

    return intersection_area(
        bbox1,
        bbox2,
    ) > 0.0


def contains_bbox(
    outer_bbox: list[float] | tuple[float, float, float, float],
    inner_bbox: list[float] | tuple[float, float, float, float],
) -> bool:
    """
    Return True if one bounding box completely
    contains another.
    """

    if not _is_valid_bbox(outer_bbox):
        raise ValueError("Invalid outer bounding box.")

    if not _is_valid_bbox(inner_bbox):
        raise ValueError("Invalid inner bounding box.")

    return (
        inner_bbox[0] >= outer_bbox[0]
        and inner_bbox[1] >= outer_bbox[1]
        and inner_bbox[2] <= outer_bbox[2]
        and inner_bbox[3] <= outer_bbox[3]
    )
# =============================================================================
# Bounding Box Utility Operations
# =============================================================================


def clip_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Clip a Pascal VOC bounding box so that it lies completely
    inside the image.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, y1, x2, y2 = bbox

    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_height))

    return (
        x1,
        y1,
        x2,
        y2,
    )


def bbox_inside_image(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> bool:
    """
    Return True if the bounding box lies completely inside the image.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, y1, x2, y2 = bbox

    return (
        x1 >= 0
        and y1 >= 0
        and x2 <= image_width
        and y2 <= image_height
    )


# =============================================================================
# Distance Utilities
# =============================================================================


def euclidean_distance(
    point1: tuple[float, float],
    point2: tuple[float, float],
) -> float:
    """
    Compute Euclidean distance between two points.
    """

    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def center_distance(
    bbox1: list[float] | tuple[float, float, float, float],
    bbox2: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Compute the distance between bounding-box centers.
    """

    center1 = bbox_center(bbox1)
    center2 = bbox_center(bbox2)

    return euclidean_distance(
        center1,
        center2,
    )


def bbox_diagonal(
    bbox: list[float] | tuple[float, float, float, float],
) -> float:
    """
    Return the diagonal length of a bounding box.
    """

    return math.sqrt(
        bbox_width(bbox) ** 2
        + bbox_height(bbox) ** 2
    )


# =============================================================================
# Bounding Box Transformations
# =============================================================================


def scale_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float]:
    """
    Scale a Pascal VOC bounding box.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, y1, x2, y2 = bbox

    return (
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    )


def translate_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    dx: float,
    dy: float,
) -> tuple[float, float, float, float]:
    """
    Translate a Pascal VOC bounding box.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, y1, x2, y2 = bbox

    return (
        x1 + dx,
        y1 + dy,
        x2 + dx,
        y2 + dy,
    )


def expand_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float, float, float]:
    """
    Expand a bounding box equally in all directions.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, y1, x2, y2 = bbox

    return (
        x1 - margin,
        y1 - margin,
        x2 + margin,
        y2 + margin,
    )


def shrink_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float, float, float]:
    """
    Shrink a bounding box equally from all sides.
    """

    if not _is_valid_bbox(bbox):
        raise ValueError("Invalid bounding box.")

    x1, y1, x2, y2 = bbox

    return (
        x1 + margin,
        y1 + margin,
        x2 - margin,
        y2 - margin,
    )
# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Internal validation helper
    "_is_valid_bbox",

    # Bounding-box measurements
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_center",

    # Format conversion
    "voc_to_yolo",
    "yolo_to_voc",
    "convert_bbox",

    # Geometry operations
    "intersection_area",
    "union_area",
    "calculate_iou",
    "overlap_ratio",
    "boxes_intersect",
    "contains_bbox",

    # Bounding-box utilities
    "clip_bbox",
    "bbox_inside_image",

    # Distance utilities
    "euclidean_distance",
    "center_distance",
    "bbox_diagonal",

    # Transformations
    "scale_bbox",
    "translate_bbox",
    "expand_bbox",
    "shrink_bbox",
]