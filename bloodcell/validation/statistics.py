"""
BloodCellAI Framework

File:
    statistics.py

Description
-----------
Dataset statistics computation for validation, reporting,
and BloodCell Dataset Quality Index (BDQI).

Version:
    1.0.0
"""

from __future__ import annotations

from collections import Counter

from .utils import average


class DatasetStatistics:
    """
    Computes dataset-level statistics.
    """

    def __init__(self):

        self.total_images = 0
        self.total_objects = 0

        self.class_counts = Counter()

        self.dataset_counts = Counter()

        self.image_widths = []

        self.image_heights = []

    ###########################################################################
    # Main Function
    ###########################################################################

    def compute(self, dataset):
        """
        Compute all statistics.
        """

        self.reset()

        if dataset is None:
            return self

        images = getattr(dataset, "images", [])

        self.total_images = len(images)

        for image in images:

            self.dataset_counts[image.dataset] += 1

            width = getattr(image, "width", None)
            height = getattr(image, "height", None)

            if width is not None:
                self.image_widths.append(width)

            if height is not None:
                self.image_heights.append(height)

            for obj in getattr(image, "objects", []) or []:

                self.total_objects += 1

                self.class_counts[obj.class_name] += 1

        return self

    ###########################################################################
    # Reset
    ###########################################################################

    def reset(self):

        self.total_images = 0

        self.total_objects = 0

        self.class_counts.clear()

        self.dataset_counts.clear()

        self.image_widths.clear()

        self.image_heights.clear()

    ###########################################################################
    # Derived Statistics
    ###########################################################################

    @property
    def objects_per_image(self):

        return average(
            self.total_objects,
            self.total_images,
        )

    @property
    def number_of_classes(self):

        return len(self.class_counts)

    @property
    def average_width(self):

        return average(
            sum(self.image_widths),
            len(self.image_widths),
        )

    @property
    def average_height(self):

        return average(
            sum(self.image_heights),
            len(self.image_heights),
        )

    ###########################################################################
    # Export
    ###########################################################################

    def to_dict(self):

        return {

            "total_images": self.total_images,

            "total_objects": self.total_objects,

            "objects_per_image": self.objects_per_image,

            "number_of_classes": self.number_of_classes,

            "average_width": self.average_width,

            "average_height": self.average_height,

            "class_counts": dict(self.class_counts),

            "dataset_counts": dict(self.dataset_counts),

        }

    ###########################################################################
    # String Representation
    ###########################################################################

    def __repr__(self):

        return (
            f"DatasetStatistics("
            f"images={self.total_images}, "
            f"objects={self.total_objects}, "
            f"classes={self.number_of_classes})"
        )