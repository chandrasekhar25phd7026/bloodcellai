"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    compose.py

Version:
    1.0.0

Description
-----------
Sequential preprocessing pipeline.

Responsibilities
----------------
✓ Execute transforms sequentially
✓ Maintain execution order
✓ Support dynamic pipelines
✓ Collect transform history
✓ Produce PreprocessingResult

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from typing import Iterable

from .base_transform import BaseTransform
from preprocessing.preprocessing_models import (
    PreprocessingResult,
)


# =============================================================================
# Compose
# =============================================================================

class Compose:
    """
    Sequential preprocessing pipeline.

    Parameters
    ----------
    transforms
        Ordered list of preprocessing transforms.
    """

    def __init__(
        self,
        transforms: Iterable[BaseTransform],
    ) -> None:

        self._transforms = list(transforms)

        self._validate()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self) -> None:
        """
        Validate transform list.
        """

        for transform in self._transforms:

            if not isinstance(
                transform,
                BaseTransform,
            ):
                raise TypeError(

                    f"{transform} is not a BaseTransform."

                )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def transforms(self) -> list[BaseTransform]:
        """
        Return pipeline transforms.
        """

        return self._transforms

    @property
    def count(self) -> int:
        """
        Number of transforms.
        """

        return len(
            self._transforms
        )
    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def __call__(
        self,
        image,
    ) -> PreprocessingResult:
        """
        Execute the preprocessing pipeline.

        Parameters
        ----------
        image
            Input image.

        Returns
        -------
        PreprocessingResult
        """

        result = PreprocessingResult()

        result.original_image = image

        current_image = image

        for transform in self._transforms:

            current_image, record = transform(current_image)

            result.add_transform(record)

            # Propagate quality metrics out of any transform that
            # computes them (QualityTransform) -- confirmed necessary
            # by actually running the pipeline: without this,
            # result.quality_metrics silently stayed at its all-zero
            # default forever, even though the transform itself
            # computed real values internally and just never had
            # anywhere to put them.
            transform_metrics = getattr(transform, "quality_metrics", None)

            if transform_metrics is not None:
                result.quality_metrics = transform_metrics

            if record.failed:

                result.add_warning(
                    f"{transform.name} failed: "
                    f"{record.message}"
                )

                if (
                    transform.config.pipeline.continue_on_failure
                    is False
                ):
                    break

        result.processed_image = current_image

        return result
    # -------------------------------------------------------------------------
    # Pipeline Management
    # -------------------------------------------------------------------------

    def append(
        self,
        transform: BaseTransform,
    ) -> None:
        """
        Append a transform to the pipeline.
        """

        if not isinstance(transform, BaseTransform):
            raise TypeError(
                "Transform must inherit from BaseTransform."
            )

        self._transforms.append(transform)

    def insert(
        self,
        index: int,
        transform: BaseTransform,
    ) -> None:
        """
        Insert a transform into the pipeline.
        """

        if not isinstance(transform, BaseTransform):
            raise TypeError(
                "Transform must inherit from BaseTransform."
            )

        self._transforms.insert(index, transform)

    def remove(
        self,
        name: str,
    ) -> None:
        """
        Remove a transform by name.
        """

        self._transforms = [

            transform

            for transform in self._transforms

            if transform.name.lower() != name.lower()

        ]

    def clear(self) -> None:
        """
        Remove all transforms.
        """

        self._transforms.clear()

    # -------------------------------------------------------------------------
    # Container Support
    # -------------------------------------------------------------------------

    def __len__(self) -> int:

        return len(self._transforms)

    def __iter__(self):

        return iter(self._transforms)

    def __getitem__(
        self,
        index: int,
    ) -> BaseTransform:

        return self._transforms[index]

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:

        names = [

            transform.name

            for transform in self._transforms

        ]

        return (

            f"{self.__class__.__name__}"

            f"(transforms={names})"

        )

    def __str__(self) -> str:

        if not self._transforms:
            return "Compose([])"

        pipeline = " -> ".join(

            transform.name

            for transform in self._transforms

        )

        return f"Compose({pipeline})"