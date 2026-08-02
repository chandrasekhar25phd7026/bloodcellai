"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    quality.py

Version:
    1.0.0

Description
-----------
Image quality assessment transform for BloodCellAI.

Computed Metrics
----------------
✓ Brightness
✓ Contrast
✓ Blur Score
✓ Sharpness
✓ Noise Level
✓ Entropy
✓ Dynamic Range
✓ Overall Quality Score

Features
--------
✓ Medical image quality assessment
✓ Automatic quality scoring
✓ Poor-quality image detection
✓ Threshold-based validation

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing_config import PreprocessingConfig
from .preprocessing_models import ImageQualityMetrics
from transforms.base_transform import BaseTransform
from transforms.registry import TransformRegistry


# =============================================================================
# Quality Transform
# =============================================================================

class QualityTransform(BaseTransform):
    """
    Image quality assessment transform.
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__(config)

        self._quality_config = config.quality

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def parameters(self) -> dict:
        """
        Return transform parameters.
        """

        return {
            "minimum_brightness":
                self._quality_config.minimum_brightness,

            "maximum_brightness":
                self._quality_config.maximum_brightness,

            "minimum_contrast":
                self._quality_config.minimum_contrast,

            "minimum_sharpness":
                self._quality_config.minimum_sharpness,

            "maximum_blur":
                self._quality_config.maximum_blur,

            "minimum_entropy":
                self._quality_config.minimum_entropy,

            "minimum_quality_score":
                self._quality_config.minimum_quality_score,

            "reject_low_quality_images":
                self._quality_config.reject_low_quality_images,
        }

    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Compute image quality metrics.

        This transform does not modify the image.
        """

        self._metrics = ImageQualityMetrics()

        gray = self._to_grayscale(image)

        self._metrics.brightness = self._brightness(gray)
        self._metrics.contrast = self._contrast(gray)
        self._metrics.dynamic_range = self._dynamic_range(gray)
        self._metrics.entropy = self._entropy(gray)

        self._metrics.blur_score = self._blur_score(gray)
        self._metrics.sharpness = self._sharpness(gray)
        self._metrics.noise_level = self._noise_level(gray)

        self._metrics.quality_score = self._quality_score()

        self._evaluate_quality()

        return image

    # -------------------------------------------------------------------------
    # Brightness
    # -------------------------------------------------------------------------

    def _brightness(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute average image brightness.
        """

        return float(
            np.mean(image)
        )

    # -------------------------------------------------------------------------
    # Contrast
    # -------------------------------------------------------------------------

    def _contrast(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute image contrast.
        """

        return float(
            np.std(image)
        )

    # -------------------------------------------------------------------------
    # Dynamic Range
    # -------------------------------------------------------------------------

    def _dynamic_range(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute image dynamic range.
        """

        return float(
            np.max(image)
            - np.min(image)
        )

    # -------------------------------------------------------------------------
    # Entropy
    # -------------------------------------------------------------------------

    def _entropy(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute image entropy.
        """

        histogram = cv2.calcHist(
            [image],
            [0],
            None,
            [256],
            [0, 256],
        )

        histogram = histogram.ravel()

        probability = (
            histogram
            / np.sum(histogram)
        )

        probability = probability[
            probability > 0
        ]

        entropy = -np.sum(
            probability
            * np.log2(probability)
        )

        return float(entropy)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _to_grayscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert image to grayscale if required.
        """

        if image.ndim == 2:
            return image

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
   
    # -------------------------------------------------------------------------
    # Blur Score
    # -------------------------------------------------------------------------

    def _blur_score(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute blur score using the variance of the Laplacian.

        Higher values indicate a sharper image.
        """

        laplacian = cv2.Laplacian(
            image,
            cv2.CV_32F,
        )

        return float(
            laplacian.var()
        )
    # -------------------------------------------------------------------------
    # Sharpness
    # -------------------------------------------------------------------------

    def _sharpness(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute image sharpness.

        Uses the mean gradient magnitude.
        """

        gradient_x = cv2.Sobel(
            image,
            cv2.CV_32F,
            1,
            0,
        )

        gradient_y = cv2.Sobel(
            image,
            cv2.CV_32F,
            0,
            1,
        )

        magnitude = np.sqrt(
            gradient_x ** 2 +
            gradient_y ** 2
        )

        return float(
            np.mean(magnitude)
        )
    # -------------------------------------------------------------------------
    # Noise Level
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Noise Level
    # -------------------------------------------------------------------------

    def _noise_level(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Estimate image noise level.

        Uses the difference between the image and
        a Gaussian-smoothed version.
        """

        blurred = cv2.GaussianBlur(
            image,
            (3, 3),
            0,
        )

        noise = image.astype(np.float32) - blurred.astype(np.float32)

        return float(
            np.std(noise)
        )

    # -------------------------------------------------------------------------
    # Overall Quality Score
    # -------------------------------------------------------------------------

    def _quality_score(
        self,
    ) -> float:
        """
        Combine the individual quality metrics into one 0-100
        composite score.

        NOTE: this method was called by apply() but never defined
        anywhere in the class -- confirmed by actually running the
        pipeline (AttributeError on first real image). Implemented
        here using the same thresholds already defined in
        QualityConfig, so a score of 100 roughly means "comfortably
        clears every configured threshold" and it degrades smoothly
        as metrics approach/cross those thresholds, rather than
        being a hard pass/fail.
        """

        scores = []

        # Brightness: full score inside [min, max]; degrades with
        # distance outside that band.
        brightness = self._metrics.brightness
        min_b = self._quality_config.minimum_brightness
        max_b = self._quality_config.maximum_brightness

        if min_b <= brightness <= max_b:
            scores.append(100.0)
        else:
            distance = min(abs(brightness - min_b), abs(brightness - max_b))
            scores.append(max(0.0, 100.0 - distance))

        # Contrast: scaled relative to the minimum threshold.
        contrast = self._metrics.contrast
        min_c = self._quality_config.minimum_contrast
        scores.append(
            min(100.0, 100.0 * contrast / min_c) if min_c > 0 else 100.0
        )

        # Blur/sharpness: blur_score is Laplacian variance (higher =
        # sharper); scaled relative to the configured threshold.
        blur = self._metrics.blur_score
        blur_threshold = self._quality_config.maximum_blur
        scores.append(
            min(100.0, 100.0 * blur / blur_threshold) if blur_threshold > 0 else 100.0
        )

        # Entropy: scaled relative to the minimum threshold.
        entropy = self._metrics.entropy
        min_e = self._quality_config.minimum_entropy
        scores.append(
            min(100.0, 100.0 * entropy / min_e) if min_e > 0 else 100.0
        )

        return round(sum(scores) / len(scores), 2)

    # -------------------------------------------------------------------------
    # Quality Evaluation
    # -------------------------------------------------------------------------

    def _evaluate_quality(
        self,
    ) -> None:
        """
        Evaluate all quality metrics against configured thresholds.
        """

        warnings = []

        self._check_brightness(warnings)
        self._check_contrast(warnings)
        self._check_sharpness(warnings)
        self._check_blur(warnings)
        self._check_entropy(warnings)
        self._check_quality_score(warnings)

        self._metrics.metadata["warnings"] = warnings

        self._metrics.passed = len(warnings) == 0

        if (
            self._quality_config.reject_low_quality_images
            and not self._metrics.passed
        ):
            raise ValueError(
                "Image failed quality assessment."
            )
    def _check_brightness(
        self,
        warnings: list[str],
    ) -> None:
        """
        Validate brightness.
        """

        if (
            self._metrics.brightness
            < self._quality_config.minimum_brightness
        ):
            warnings.append(
                "Brightness below minimum threshold."
            )

        if (
            self._metrics.brightness
            > self._quality_config.maximum_brightness
        ):
            warnings.append(
                "Brightness above maximum threshold."
            )
    def _check_contrast(
        self,
        warnings: list[str],
    ) -> None:
        """
        Validate contrast.
        """

        if (
            self._metrics.contrast
            < self._quality_config.minimum_contrast
        ):
            warnings.append(
                "Contrast below minimum threshold."
            )
    def _check_sharpness(
        self,
        warnings: list[str],
    ) -> None:
        """
        Validate sharpness.
        """

        if (
            self._metrics.sharpness
            < self._quality_config.minimum_sharpness
        ):
            warnings.append(
                "Sharpness below minimum threshold."
            )
    def _check_blur(
        self,
        warnings: list[str],
    ) -> None:
        """
        Validate blur.
        """

        if (
            self._metrics.blur_score
            < self._quality_config.maximum_blur
        ):
            warnings.append(
                "Image appears blurry."
            )
    def _check_entropy(
        self,
        warnings: list[str],
    ) -> None:
        """
        Validate entropy.
        """

        if (
            self._metrics.entropy
            < self._quality_config.minimum_entropy
        ):
            warnings.append(
                "Entropy below minimum threshold."
            )
    def _check_quality_score(
        self,
        warnings: list[str],
    ) -> None:
        """
        Validate overall quality score.
        """

        if (
            self._metrics.quality_score
            < self._quality_config.minimum_quality_score
        ):
            warnings.append(
                "Overall quality score below minimum threshold."
            )
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def quality_metrics(
        self,
    ) -> ImageQualityMetrics:
        """
        Return the most recently computed quality metrics.
        """

        return self._metrics

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_input(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the input image.
        """

        super().validate_input(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input must be a NumPy ndarray."
            )

        if image.size == 0:
            raise ValueError(
                "Input image is empty."
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                "Input image must be either grayscale "
                "(2D) or color (3D)."
            )

    def validate_output(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the output image.

        QualityTransform does not modify the image.
        """

        super().validate_output(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Output must be a NumPy ndarray."
            )

        if image.shape != self._input_shape:
            raise ValueError(
                "Output image shape differs from input image."
            )

        if image.dtype != self._input_dtype:
            raise TypeError(
                "Output image dtype differs from input image."
            )

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    def before_apply(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Store input information before processing.
        """

        self._input_shape = image.shape
        self._input_dtype = image.dtype

    def after_apply(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Hook executed after processing.

        No post-processing required.
        """

        return None

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Reset internal state.
        """

        self._metrics = ImageQualityMetrics()
        self._input_shape = None
        self._input_dtype = None

# =============================================================================
# Register Transform
# =============================================================================

TransformRegistry.register(
    "quality",
    QualityTransform,
)
