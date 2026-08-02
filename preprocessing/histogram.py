"""
BloodCellAI
Histogram Processing Transform

Provides histogram-based image enhancement and
analysis utilities for blood smear image
preprocessing.

Supported Operations

- Histogram Computation
- Histogram Equalization
- Contrast Stretching
- Histogram Stretching
- Histogram Matching
- Gamma Correction
- Brightness Normalization
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import cv2
import numpy as np

from .preprocessing_models import (
    TransformRecord,
    TransformStatus,
)

from .preprocessing_config import (
    PreprocessingConfig,
)

from transforms.base_transform import (
    BaseTransform,
)

from transforms.registry import (
    TransformRegistry,
)


class HistogramOperation(
    Enum,
):
    """
    Supported histogram operations.
    """

    HISTOGRAM = "histogram"

    EQUALIZATION = "equalization"

    STRETCH = "stretch"

    CONTRAST_STRETCH = "contrast_stretch"

    MATCHING = "matching"

    GAMMA = "gamma"

    BRIGHTNESS_NORMALIZATION = (
        "brightness_normalization"
    )


class HistogramTransform(
    BaseTransform,
):
    """
    Histogram processing transform.

    Provides histogram analysis and
    enhancement methods commonly used
    in biomedical image preprocessing.
    """

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__()

        self._config = config

        self._logger = logging.getLogger(
            self.__class__.__name__
        )

        histogram = getattr(
            config,
            "histogram",
            None,
        )

        if histogram is None:

            raise ValueError(
                "Histogram configuration "
                "is missing."
            )

        self._enabled = histogram.enabled

        self._operation = HistogramOperation(
            histogram.operation
        )

        self._gamma = histogram.gamma

        self._clip_limit = (
            histogram.clip_limit
        )

        self._tile_grid_size = (
            histogram.tile_grid_size
        )

        self._stretch_min = (
            histogram.stretch_min
        )

        self._stretch_max = (
            histogram.stretch_max
        )

        self._reference_image = getattr(
            histogram,
            "reference_image",
            None,
        )

        self._last_histogram = None

        self._last_cdf = None

        self._lookup_table = None

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_input(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate input image.
        """

        if image is None:

            raise ValueError(
                "Image is None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):

            raise TypeError(
                "Expected numpy.ndarray."
            )

        if image.size == 0:

            raise ValueError(
                "Image is empty."
            )

        if image.ndim not in (2, 3):

            raise ValueError(
                "Only grayscale or "
                "colour images are "
                "supported."
            )

        if self._gamma <= 0:

            raise ValueError(
                "Gamma must be "
                "greater than zero."
            )

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------

    @property
    def parameters(
        self,
    ) -> dict[str, Any]:
        """
        Return transform parameters.
        """

        return {

            "enabled":
                self._enabled,

            "operation":
                self._operation.value,

            "gamma":
                self._gamma,

            "clip_limit":
                self._clip_limit,

            "tile_grid_size":
                self._tile_grid_size,

            "stretch_min":
                self._stretch_min,

            "stretch_max":
                self._stretch_max,

        }

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def operation(
        self,
    ) -> HistogramOperation:

        return self._operation


    @property
    def gamma(
        self,
    ) -> float:

        return self._gamma


    @property
    def histogram(
        self,
    ) -> np.ndarray | None:
        """
        Return the last computed histogram.
        """

        return self._last_histogram


    @property
    def cdf(
        self,
    ) -> np.ndarray | None:
        """
        Return the last computed CDF.
        """

        return self._last_cdf


    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Reset internal state.
        """

        self._last_histogram = None

        self._last_cdf = None

        self._lookup_table = None

    # ---------------------------------------------------------
    # Histogram Computation
    # ---------------------------------------------------------

    def _calculate_histogram(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the histogram of an image.

        For colour images, the image is
        converted to grayscale before
        histogram computation.
        """

        if image.ndim == 3:

            image = cv2.cvtColor(

                image,

                cv2.COLOR_BGR2GRAY,

            )

        histogram = cv2.calcHist(

            [image],

            [0],

            None,

            [256],

            [0, 256],

        )

        histogram = histogram.flatten()

        self._last_histogram = histogram

        return histogram


    # ---------------------------------------------------------
    # Histogram Normalization
    # ---------------------------------------------------------

    def _normalize_histogram(
        self,
        histogram: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize histogram values
        between 0 and 1.
        """

        histogram = histogram.astype(
            np.float32
        )

        total = histogram.sum()

        if total == 0:

            return histogram

        return histogram / total

    # ---------------------------------------------------------
    # Cumulative Distribution Function
    # ---------------------------------------------------------

    def _calculate_cdf(
        self,
        histogram: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the cumulative
        distribution function (CDF).
        """

        cdf = np.cumsum(
            histogram
        )

        self._last_cdf = cdf

        return cdf


    # ---------------------------------------------------------
    # Histogram Statistics
    # ---------------------------------------------------------

    def _histogram_statistics(
        self,
        histogram: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute histogram statistics.
        """

        histogram = self._normalize_histogram(
            histogram
        )

        intensity = np.arange(
            256,
            dtype=np.float32,
        )

        mean = np.sum(
            intensity * histogram
        )

        variance = np.sum(

            ((intensity - mean) ** 2)

            * histogram

        )

        std = np.sqrt(
            variance
        )

        return {

            "mean":
                float(mean),

            "variance":
                float(variance),

            "std":
                float(std),

            "minimum":
                float(np.min(histogram)),

            "maximum":
                float(np.max(histogram)),

        }


    # ---------------------------------------------------------
    # Lookup Table
    # ---------------------------------------------------------

    def _build_lookup_table(
        self,
        gamma: float,
    ) -> np.ndarray:
        """
        Build gamma correction
        lookup table.
        """

        inv_gamma = 1.0 / gamma

        table = np.array(

            [

                (

                    (i / 255.0)

                    ** inv_gamma

                )

                * 255

                for i in range(256)

            ],

            dtype=np.float32,

        )

        table = np.clip(

            table,

            0,

            255,

        ).astype(
            np.uint8
        )

        self._lookup_table = table

        return table


    # ---------------------------------------------------------
    # Grayscale Conversion
    # ---------------------------------------------------------

    def _to_grayscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert image to grayscale.
        """

        if image.ndim == 2:

            return image

        return cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY,

        )


    # ---------------------------------------------------------
    # Image Statistics
    # ---------------------------------------------------------

    def _image_statistics(
        self,
        image: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute image intensity
        statistics.
        """

        gray = self._to_grayscale(
            image
        )

        return {

            "minimum":
                float(
                    np.min(gray)
                ),

            "maximum":
                float(
                    np.max(gray)
                ),

            "mean":
                float(
                    np.mean(gray)
                ),

            "median":
                float(
                    np.median(gray)
                ),

            "std":
                float(
                    np.std(gray)
                ),

        }


    # ---------------------------------------------------------
    # Histogram Summary
    # ---------------------------------------------------------

    @property
    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return histogram transform
        summary.
        """

        summary = {

            "transform":

                "Histogram",

            "enabled":

                self._enabled,

            "operation":

                self._operation.value,

            "gamma":

                self._gamma,

            "clip_limit":

                self._clip_limit,

            "tile_grid_size":

                self._tile_grid_size,

            "stretch_min":

                self._stretch_min,

            "stretch_max":

                self._stretch_max,

        }

        if self._last_histogram is not None:

            summary[
                "statistics"
            ] = self._histogram_statistics(

                self._last_histogram

            )

        return summary
    # ---------------------------------------------------------
    # Histogram Equalization
    # ---------------------------------------------------------

    def _equalize_histogram(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Perform histogram equalization.

        For colour images only the luminance
        channel is equalized in LAB colour
        space to preserve colours.
        """

        if image.ndim == 2:

            return cv2.equalizeHist(
                image
            )

        lab = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2LAB,

        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        l_channel = cv2.equalizeHist(
            l_channel
        )

        merged = cv2.merge(

            (
                l_channel,
                a_channel,
                b_channel,
            )

        )

        return cv2.cvtColor(

            merged,

            cv2.COLOR_LAB2BGR,

        )


    # ---------------------------------------------------------
    # CLAHE Enhancement
    # ---------------------------------------------------------

    def _clahe(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Contrast Limited
        Adaptive Histogram
        Equalization.
        """

        clahe = cv2.createCLAHE(

            clipLimit=self._clip_limit,

            tileGridSize=(
                self._tile_grid_size,
                self._tile_grid_size,
            ),

        )

        if image.ndim == 2:

            return clahe.apply(
                image
            )

        lab = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2LAB,

        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        l_channel = clahe.apply(
            l_channel
        )

        merged = cv2.merge(

            (
                l_channel,
                a_channel,
                b_channel,
            )

        )

        return cv2.cvtColor(

            merged,

            cv2.COLOR_LAB2BGR,

        )


    # ---------------------------------------------------------
    # Contrast Stretching
    # ---------------------------------------------------------

    def _contrast_stretch(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Stretch intensity values
        between configured limits.
        """

        minimum = float(
            self._stretch_min
        )

        maximum = float(
            self._stretch_max
        )

        if maximum <= minimum:

            raise ValueError(
                "stretch_max must be "
                "greater than stretch_min."
            )

        image_float = image.astype(
            np.float32
        )

        image_float = (

            image_float - minimum

        ) / (

            maximum - minimum

        )

        image_float = np.clip(

            image_float,

            0.0,

            1.0,

        )

        image_float *= 255.0

        return image_float.astype(
            np.uint8
        )


    # ---------------------------------------------------------
    # Histogram Stretching
    # ---------------------------------------------------------

    def _histogram_stretch(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Stretch image histogram
        using image minimum and
        maximum intensities.
        """

        image_float = image.astype(
            np.float32
        )

        minimum = np.min(
            image_float
        )

        maximum = np.max(
            image_float
        )

        if maximum == minimum:

            return image.copy()

        stretched = (

            image_float - minimum

        ) / (

            maximum - minimum

        )

        stretched *= 255.0

        return stretched.astype(
            np.uint8
        )


    # ---------------------------------------------------------
    # Gamma Correction
    # ---------------------------------------------------------

    def _gamma_correction(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply gamma correction
        using a lookup table.
        """

        table = self._lookup_table

        if table is None:

            table = self._build_lookup_table(
                self._gamma
            )

        return cv2.LUT(

            image,

            table,

        )
    # ---------------------------------------------------------
    # Brightness Normalization
    # ---------------------------------------------------------

    def _brightness_normalization(
        self,
        image: np.ndarray,
        target_mean: float = 128.0,
    ) -> np.ndarray:
        """
        Normalize image brightness to a
        desired mean intensity.
        """

        image_float = image.astype(
            np.float32
        )

        current_mean = np.mean(
            image_float
        )

        if current_mean <= 0:

            return image.copy()

        scale = (
            target_mean /
            current_mean
        )

        normalized = (
            image_float * scale
        )

        normalized = np.clip(

            normalized,

            0,

            255,

        )

        return normalized.astype(
            np.uint8
        )


    # ---------------------------------------------------------
    # Histogram Matching Utilities
    # ---------------------------------------------------------

    def _calculate_reference_histogram(
        self,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate normalized histogram of
        the reference image.
        """

        if reference.ndim == 3:

            reference = cv2.cvtColor(

                reference,

                cv2.COLOR_BGR2GRAY,

            )

        histogram = cv2.calcHist(

            [reference],

            [0],

            None,

            [256],

            [0, 256],

        ).flatten()

        histogram = self._normalize_histogram(
            histogram
        )

        return histogram


    def _calculate_reference_cdf(
        self,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cumulative distribution
        function of reference image.
        """

        histogram = (
            self._calculate_reference_histogram(
                reference
            )
        )

        return np.cumsum(
            histogram
        )


    def _create_histogram_mapping(
        self,
        source_cdf: np.ndarray,
        reference_cdf: np.ndarray,
    ) -> np.ndarray:
        """
        Create intensity mapping between
        source and reference cumulative
        distributions.
        """

        mapping = np.zeros(

            256,

            dtype=np.uint8,

        )

        reference_index = 0

        for source_index in range(256):

            while (

                reference_index < 255

                and

                reference_cdf[
                    reference_index
                ]

                <

                source_cdf[
                    source_index
                ]

            ):

                reference_index += 1

            mapping[
                source_index
            ] = reference_index

        return mapping


    def _apply_histogram_mapping(
        self,
        image: np.ndarray,
        mapping: np.ndarray,
    ) -> np.ndarray:
        """
        Apply histogram mapping using
        a lookup table.
        """

        if image.ndim == 2:

            return cv2.LUT(

                image,

                mapping,

            )

        lab = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2LAB,

        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        l_channel = cv2.LUT(

            l_channel,

            mapping,

        )

        merged = cv2.merge(

            (

                l_channel,

                a_channel,

                b_channel,

            )

        )

        return cv2.cvtColor(

            merged,

            cv2.COLOR_LAB2BGR,

        )


    # ---------------------------------------------------------
    # Histogram Matching Preparation
    # ---------------------------------------------------------

    def _prepare_histogram_matching(
        self,
        image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Prepare source histogram, source
        CDF and reference CDF for
        histogram matching.
        """

        if self._reference_image is None:

            raise ValueError(

                "Reference image is "
                "required for histogram "
                "matching."

            )

        source_histogram = (
            self._calculate_histogram(
                image
            )
        )

        source_histogram = (
            self._normalize_histogram(
                source_histogram
            )
        )

        source_cdf = np.cumsum(
            source_histogram
        )

        reference_cdf = (
            self._calculate_reference_cdf(
                self._reference_image
            )
        )

        return (

            source_histogram,

            source_cdf,

            reference_cdf,

        )
    # ---------------------------------------------------------
    # Histogram Matching
    # ---------------------------------------------------------

    def _histogram_matching(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Match the histogram of the input
        image with the reference image.
        """

        (
            _,
            source_cdf,
            reference_cdf,
        ) = self._prepare_histogram_matching(
            image
        )

        mapping = self._create_histogram_mapping(
            source_cdf,
            reference_cdf,
        )

        return self._apply_histogram_mapping(
            image,
            mapping,
        )


    # ---------------------------------------------------------
    # Operation Dispatcher
    # ---------------------------------------------------------

    def _apply_operation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the configured histogram
        operation.
        """

        if not self._enabled:

            return image.copy()

        operation = self._operation

        if operation == HistogramOperation.HISTOGRAM:

            self._calculate_histogram(
                image
            )

            return image.copy()

        if operation == HistogramOperation.EQUALIZATION:

            return self._equalize_histogram(
                image
            )

        if operation == HistogramOperation.STRETCH:

            return self._histogram_stretch(
                image
            )

        if operation == HistogramOperation.CONTRAST_STRETCH:

            return self._contrast_stretch(
                image
            )

        if operation == HistogramOperation.MATCHING:

            return self._histogram_matching(
                image
            )

        if operation == HistogramOperation.GAMMA:

            return self._gamma_correction(
                image
            )

        if (
            operation
            ==
            HistogramOperation.BRIGHTNESS_NORMALIZATION
        ):

            return self._brightness_normalization(
                image
            )

        raise ValueError(

            f"Unsupported histogram "
            f"operation: "
            f"{operation}"

        )


    # ---------------------------------------------------------
    # Operation Validation
    # ---------------------------------------------------------

    def _validate_operation(
        self,
    ) -> None:
        """
        Validate histogram operation
        configuration.
        """

        if not isinstance(
            self._operation,
            HistogramOperation,
        ):

            raise TypeError(
                "Invalid histogram "
                "operation."
            )

        if (
            self._operation
            ==
            HistogramOperation.GAMMA
        ):

            if self._gamma <= 0:

                raise ValueError(
                    "Gamma must be "
                    "greater than zero."
                )

        if (
            self._operation
            ==
            HistogramOperation.CONTRAST_STRETCH
        ):

            if (
                self._stretch_max
                <=
                self._stretch_min
            ):

                raise ValueError(
                    "stretch_max must "
                    "be greater than "
                    "stretch_min."
                )

        if (
            self._operation
            ==
            HistogramOperation.MATCHING
        ):

            if self._reference_image is None:

                raise ValueError(
                    "Reference image "
                    "is required for "
                    "histogram matching."
                )

        if (
            self._operation
            ==
            HistogramOperation.HISTOGRAM
        ):

            return

        return
    # ---------------------------------------------------------
    # Before Apply
    # ---------------------------------------------------------

    def before_apply(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Execute pre-processing steps before
        applying the histogram operation.
        """

        self.validate_input(
            image
        )

        self._validate_operation()

        self._logger.debug(

            "Starting histogram "
            "operation: %s",

            self._operation.value,

        )

        self._last_histogram = None

        self._last_cdf = None

        if (

            self._operation
            ==
            HistogramOperation.GAMMA

        ):

            if self._lookup_table is None:

                self._lookup_table = (

                    self._build_lookup_table(

                        self._gamma

                    )

                )


    # ---------------------------------------------------------
    # Apply
    # ---------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the configured histogram
        transformation.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        numpy.ndarray
            Processed image.
        """

        if not self._enabled:

            self._logger.debug(

                "Histogram transform "
                "is disabled."

            )

            return image.copy()

        self._logger.debug(

            "Executing histogram "
            "operation '%s'.",

            self._operation.value,

        )

        output = self._apply_operation(
            image
        )

        if self._last_histogram is None:

            self._calculate_histogram(
                output
            )

        if self._last_cdf is None:

            normalized = (

                self._normalize_histogram(

                    self._last_histogram

                )

            )

            self._last_cdf = (

                self._calculate_cdf(

                    normalized

                )

            )

        self._logger.debug(

            "Histogram operation "
            "completed successfully."

        )

        return output
    # ---------------------------------------------------------
    # Validate Output
    # ---------------------------------------------------------

    def validate_output(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the processed image.
        """

        if image is None:

            raise ValueError(
                "Processed image is None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):

            raise TypeError(
                "Processed image must "
                "be a numpy.ndarray."
            )

        if image.size == 0:

            raise ValueError(
                "Processed image is empty."
            )

        if image.ndim not in (2, 3):

            raise ValueError(
                "Processed image has "
                "an invalid number of "
                "dimensions."
            )

        if image.dtype != np.uint8:

            raise TypeError(
                "Processed image must "
                "have uint8 data type."
            )

        minimum = np.min(
            image
        )

        maximum = np.max(
            image
        )

        if minimum < 0 or maximum > 255:

            raise ValueError(
                "Pixel values must "
                "remain within the "
                "range [0, 255]."
            )


    # ---------------------------------------------------------
    # After Apply
    # ---------------------------------------------------------

    def after_apply(
        self,
        image: np.ndarray,
    ) -> TransformRecord:
        """
        Execute post-processing after the
        histogram transformation and
        create a TransformRecord.
        """

        histogram = self._last_histogram

        if histogram is None:

            histogram = self._calculate_histogram(
                image
            )

        normalized = self._normalize_histogram(
            histogram
        )

        cdf = self._calculate_cdf(
            normalized
        )

        statistics = self._histogram_statistics(
            histogram
        )

        image_statistics = self._image_statistics(
            image
        )

        metadata = {

            "operation":
                self._operation.value,

            "histogram_statistics":
                statistics,

            "image_statistics":
                image_statistics,

            "cdf_available":
                cdf is not None,

            "gamma":
                self._gamma,

            "clip_limit":
                self._clip_limit,

            "tile_grid_size":
                self._tile_grid_size,

            "stretch_min":
                self._stretch_min,

            "stretch_max":
                self._stretch_max,

        }

        record = TransformRecord(

            name="Histogram",

            status=TransformStatus.SUCCESS,

            parameters=self.parameters,

            metadata=metadata,

        )

        self._logger.debug(

            "Histogram transform "
            "completed successfully."

        )

        return record
    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(
        self,
    ) -> str:
        """
        Developer representation.
        """

        return (

            f"{self.__class__.__name__}("

            f"enabled={self._enabled}, "

            f"operation='{self._operation.value}', "

            f"gamma={self._gamma}, "

            f"clip_limit={self._clip_limit}, "

            f"tile_grid_size={self._tile_grid_size}, "

            f"stretch_min={self._stretch_min}, "

            f"stretch_max={self._stretch_max}"

            ")"

        )


    def __str__(
        self,
    ) -> str:
        """
        Human readable summary.
        """

        return (

            "HistogramTransform\n"

            f"  Enabled           : {self._enabled}\n"

            f"  Operation         : {self._operation.value}\n"

            f"  Gamma             : {self._gamma}\n"

            f"  Clip Limit        : {self._clip_limit}\n"

            f"  Tile Grid Size    : {self._tile_grid_size}\n"

            f"  Stretch Minimum   : {self._stretch_min}\n"

            f"  Stretch Maximum   : {self._stretch_max}"

        )


# ---------------------------------------------------------
# Transform Registration
# ---------------------------------------------------------

TransformRegistry.register(

    "histogram",

    HistogramTransform,

)


# ---------------------------------------------------------
# Module Information
# ---------------------------------------------------------

__all__ = [

    "HistogramOperation",

    "HistogramTransform",

]


"""
============================================================
Usage Examples
============================================================

Example 1
---------

config.histogram.enabled = True

config.histogram.operation = "equalization"

transform = HistogramTransform(config)

image, record = transform(image)



Example 2
---------

config.histogram.operation = "gamma"

config.histogram.gamma = 1.5

transform = HistogramTransform(config)

image, record = transform(image)



Example 3
---------

config.histogram.operation = "stretch"

transform = HistogramTransform(config)

image, record = transform(image)



Example 4
---------

config.histogram.operation = "contrast_stretch"

config.histogram.stretch_min = 15

config.histogram.stretch_max = 240

transform = HistogramTransform(config)

image, record = transform(image)



Example 5
---------

config.histogram.operation = "matching"

config.histogram.reference_image = reference

transform = HistogramTransform(config)

image, record = transform(image)



Example 6
---------

config.histogram.operation = "brightness_normalization"

transform = HistogramTransform(config)

image, record = transform(image)



Example 7
---------

config.histogram.operation = "histogram"

transform = HistogramTransform(config)

image, record = transform(image)

histogram = transform.histogram

cdf = transform.cdf

statistics = transform.summary


============================================================
End of histogram.py
============================================================
