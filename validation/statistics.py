"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    statistics.py

Version:
    4.0.0

Purpose:
    Dataset Statistics Engine

Description:
    Computes descriptive statistics for a UniversalDataset after
    validation has completed.

The statistics engine is independent from the validator and may be
used separately for reporting and dataset analysis.

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable

from .models import ValidationStatistics

from bloodcell.universal_dataset import UniversalDataset
from bloodcell.universal_object import (
    UniversalImage,
    BoundingBox,
)
# =============================================================================
# Dataset Statistics Engine
# =============================================================================


class DatasetStatisticsEngine:
    """
    Computes descriptive statistics for a UniversalDataset.

    Responsibilities
    ----------------
    ✓ Dataset counts
    ✓ Image statistics
    ✓ Object statistics
    ✓ Bounding-box statistics
    ✓ Dataset distribution
    ✓ Class distribution
    ✓ Split distribution

    This class performs no validation.
    """

    def __init__(self):

        self.statistics = ValidationStatistics()
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _iter_images(
        self,
        dataset: UniversalDataset,
    ) -> Iterable[UniversalImage]:

        yield from dataset.images


    def _iter_objects(
        self,
        image: UniversalImage,
    ) -> Iterable[BoundingBox]:

        yield from image.objects


    def reset(self):

        self.statistics.reset()
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def compute(
        self,
        dataset: UniversalDataset,
    ) -> ValidationStatistics:
        """
        Compute statistics for a UniversalDataset.

        Parameters
        ----------
        dataset : UniversalDataset

        Returns
        -------
        ValidationStatistics
        """

        self.reset()

        self._dataset_counts(dataset)

        self._image_statistics(dataset)

        self._object_statistics(dataset)

        self._bounding_box_statistics(dataset)

        self._distribution_statistics(dataset)

        return self.statistics
    # -------------------------------------------------------------------------
    # Dataset Counts
    # -------------------------------------------------------------------------

    def _dataset_counts(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Compute dataset-level counts.
        """

        stats = self.statistics

        stats.total_images = len(dataset.images)

        stats.total_objects = sum(
            len(image.objects)
            for image in dataset.images
        )

        stats.total_bounding_boxes = stats.total_objects

    # -------------------------------------------------------------------------
    # Image Statistics
    # -------------------------------------------------------------------------

    def _image_statistics(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Compute image dimension statistics.
        """

        stats = self.statistics

        images = dataset.images

        if not images:
            return

        widths = [
            image.width
            for image in images
            if image.width is not None
        ]

        heights = [
            image.height
            for image in images
            if image.height is not None
        ]

        # ---------------------------------------------------------
        # Width Statistics
        # ---------------------------------------------------------

        if widths:

            stats.minimum_width = min(widths)

            stats.maximum_width = max(widths)

            stats.average_width = mean(widths)

        # ---------------------------------------------------------
        # Height Statistics
        # ---------------------------------------------------------

        if heights:

            stats.minimum_height = min(heights)

            stats.maximum_height = max(heights)

            stats.average_height = mean(heights)

        # ---------------------------------------------------------
        # Image Formats
        # ---------------------------------------------------------

        image_formats = Counter()

        for image in images:

            image_path = str(image.image_path)

            if "." not in image_path:
                continue

            extension = (
                image_path
                .split(".")[-1]
                .lower()
            )

            image_formats[extension] += 1

        stats.image_formats = dict(image_formats)
    # -------------------------------------------------------------------------
    # Object Statistics
    # -------------------------------------------------------------------------

    def _object_statistics(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Compute object-level statistics.

        Calculates:

        - objects per image
        - minimum objects per image
        - maximum objects per image
        - average objects per image
        - empty images
        """

        stats = self.statistics

        object_counts = []

        empty_images = 0

        for image in dataset.images:

            count = len(image.objects)

            object_counts.append(count)

            if count == 0:
                empty_images += 1

        if not object_counts:

            stats.objects_per_image = []

            stats.minimum_objects_per_image = 0

            stats.maximum_objects_per_image = 0

            stats.empty_images = 0

            return

        # ---------------------------------------------------------
        # Store counts
        # ---------------------------------------------------------

        stats.objects_per_image = object_counts

        # ---------------------------------------------------------
        # Minimum / Maximum
        # ---------------------------------------------------------

        stats.minimum_objects_per_image = min(
            object_counts
        )

        stats.maximum_objects_per_image = max(
            object_counts
        )

        # ---------------------------------------------------------
        # Average
        # ---------------------------------------------------------

        # ValidationStatistics computes the average using
        # objects_per_image, so no assignment is required.

        # ---------------------------------------------------------
        # Empty Images
        # ---------------------------------------------------------

        stats.empty_images = empty_images
    # -------------------------------------------------------------------------
    # Bounding Box Statistics
    # -------------------------------------------------------------------------

    def _bounding_box_statistics(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Compute bounding-box statistics.

        Calculates:

        - minimum_bbox_width
        - maximum_bbox_width
        - average_bbox_width

        - minimum_bbox_height
        - maximum_bbox_height
        - average_bbox_height

        - minimum_bbox_area
        - maximum_bbox_area
        - average_bbox_area
        """

        stats = self.statistics

        bbox_widths = []
        bbox_heights = []
        bbox_areas = []

        for image in dataset.images:

            for bbox in image.objects:

                # -------------------------------
                # Width
                # -------------------------------

                if bbox.w is not None:
                    bbox_widths.append(float(bbox.w))

                # -------------------------------
                # Height
                # -------------------------------

                if bbox.h is not None:
                    bbox_heights.append(float(bbox.h))

                # -------------------------------
                # Area
                # -------------------------------

                if bbox.area is not None:

                    bbox_areas.append(float(bbox.area))

                elif (
                    bbox.w is not None
                    and bbox.h is not None
                ):

                    bbox_areas.append(
                        float(bbox.w) * float(bbox.h)
                    )

        # ---------------------------------------------------------
        # Width Statistics
        # ---------------------------------------------------------

        if bbox_widths:

            stats.minimum_bbox_width = min(
                bbox_widths
            )

            stats.maximum_bbox_width = max(
                bbox_widths
            )

            stats.average_bbox_width = mean(
                bbox_widths
            )

        # ---------------------------------------------------------
        # Height Statistics
        # ---------------------------------------------------------

        if bbox_heights:

            stats.minimum_bbox_height = min(
                bbox_heights
            )

            stats.maximum_bbox_height = max(
                bbox_heights
            )

            stats.average_bbox_height = mean(
                bbox_heights
            )

        # ---------------------------------------------------------
        # Area Statistics
        # ---------------------------------------------------------

        if bbox_areas:

            stats.minimum_bbox_area = min(
                bbox_areas
            )

            stats.maximum_bbox_area = max(
                bbox_areas
            )

            stats.average_bbox_area = mean(
                bbox_areas
            )
    # -------------------------------------------------------------------------
    # Distribution Statistics
    # -------------------------------------------------------------------------

    def _distribution_statistics(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Compute dataset distribution statistics.

        Computes

        - dataset_counts
        - class_counts
        - split_counts
        """

        stats = self.statistics

        # ---------------------------------------------------------
        # Dataset Distribution
        # ---------------------------------------------------------

        if getattr(dataset, "dataset_counts", None):

            stats.dataset_counts = dict(
                dataset.dataset_counts
            )

        else:

            dataset_counter = Counter()

            for image in dataset.images:

                dataset_counter[image.dataset] += 1

            stats.dataset_counts = dict(
                dataset_counter
            )

        # ---------------------------------------------------------
        # Class Distribution
        # ---------------------------------------------------------

        if getattr(dataset, "class_counts", None):

            stats.class_counts = dict(
                dataset.class_counts
            )

        else:

            class_counter = Counter()

            for image in dataset.images:

                for bbox in image.objects:

                    class_counter[bbox.class_name] += 1

            stats.class_counts = dict(
                class_counter
            )

        # ---------------------------------------------------------
        # Split Distribution
        # ---------------------------------------------------------

        split_counter = Counter()

        for image in dataset.images:

            split_counter[image.split] += 1

        stats.split_counts = dict(
            split_counter
        )
# =============================================================================
# Convenience Function
# =============================================================================

def compute_statistics(
    dataset: UniversalDataset,
) -> ValidationStatistics:
    """
    Compute statistics for a UniversalDataset.

    Parameters
    ----------
    dataset : UniversalDataset

    Returns
    -------
    ValidationStatistics
    """

    engine = DatasetStatisticsEngine()

    return engine.compute(dataset)
# =============================================================================
# Public API
# =============================================================================

__all__ = [

    "DatasetStatisticsEngine",

    "compute_statistics",

]
