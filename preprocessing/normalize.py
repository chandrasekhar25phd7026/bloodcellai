"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    normalize.py

Version:
    1.0.0

Description
-----------
Image normalization transform for the BloodCellAI preprocessing framework.

Features
--------
✓ Min-Max Normalization
✓ Unit Scale Normalization
✓ Z-Score Normalization
✓ ImageNet Normalization
✓ Custom Mean/Standard Deviation Normalization

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing_config import PreprocessingConfig
from .preprocessing_models import NormalizationMethod
from transforms.base_transform import BaseTransform
from transforms.registry import TransformRegistry


# =============================================================================
# Normalize Transform
# =============================================================================

class NormalizeTransform(BaseTransform):
    """
    Normalize an image according to the preprocessing configuration.
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__(config)

        self._normalize_config = config.normalize

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def parameters(self) -> dict:
        """
        Return transform parameters.
        """

        return {
            "method": self._normalize_config.method.value,
            "scale_min": self._normalize_config.scale_min,
            "scale_max": self._normalize_config.scale_max,
            "mean": self._normalize_config.mean,
            "std": self._normalize_config.std,
            "clip_values": self._normalize_config.clip_values,
        }
    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply image normalization.
        """

        method = self._normalize_config.method

        if method == NormalizationMethod.NONE:
            return image.copy()

        if method == NormalizationMethod.MIN_MAX:
            return self._min_max_normalize(image)

        if method == NormalizationMethod.UNIT_SCALE:
            return self._unit_scale_normalize(image)

        if method == NormalizationMethod.Z_SCORE:
            return self._z_score_normalize(image)

        if method == NormalizationMethod.IMAGENET:
            return self._imagenet_normalize(image)

        if method == NormalizationMethod.CUSTOM:
            return self._custom_normalize(image)

        raise ValueError(
            f"Unsupported normalization method: {method}"
        )

    # -------------------------------------------------------------------------
    # Min-Max Normalization
    # -------------------------------------------------------------------------

    def _min_max_normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Scale image to the configured range.
        """

        image = image.astype(np.float32)

        minimum = image.min()
        maximum = image.max()

        if maximum == minimum:
            return self._clip(image)

        image = (image - minimum) / (maximum - minimum)

        image = (
            image
            * (
                self._normalize_config.scale_max
                - self._normalize_config.scale_min
            )
            + self._normalize_config.scale_min
        )

        return self._clip(image)

    # -------------------------------------------------------------------------
    # Unit Scale Normalization
    # -------------------------------------------------------------------------

    def _unit_scale_normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Scale pixel values from [0,255] to [0,1].
        """

        image = image.astype(np.float32) / 255.0

        return self._clip(image)

    # -------------------------------------------------------------------------
    # Z-Score Normalization
    # -------------------------------------------------------------------------

    def _z_score_normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Standard score normalization.
        """

        image = image.astype(np.float32)

        mean = np.mean(image)
        std = np.std(image)

        if std == 0:
            return image

        image = (image - mean) / std
        return self._clip(image)

    # -------------------------------------------------------------------------
    # ImageNet Normalization
    # -------------------------------------------------------------------------

    def _imagenet_normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize using ImageNet mean and standard deviation.
        """

        image = image.astype(np.float32) / 255.0

        mean = np.array(
            self._normalize_config.mean,
            dtype=np.float32,
        )

        std = np.array(
            self._normalize_config.std,
            dtype=np.float32,
        )

        image = (image - mean) / std

        return self._clip(image)

    # -------------------------------------------------------------------------
    # Custom Normalization
    # -------------------------------------------------------------------------

    def _custom_normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize using user-defined mean and standard deviation.
        """

        image = image.astype(np.float32)

        mean = np.array(
            self._normalize_config.mean,
            dtype=np.float32,
        )

        std = np.array(
            self._normalize_config.std,
            dtype=np.float32,
        )

        std = np.where(std == 0, 1.0, std)

        image = (image - mean) / std

        return self._clip(image)
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
        Validate the normalized image.
        """

        super().validate_output(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Output must be a NumPy ndarray."
            )

        if image.dtype not in (
            np.float32,
            np.float64,
        ):
            raise TypeError(
                "Normalized image must have a floating-point dtype."
            )

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _clip(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Clip normalized values if enabled.
        """

        if not self._normalize_config.clip_values:
            return image

        return np.clip(
            image,
            self._normalize_config.scale_min,
            self._normalize_config.scale_max,
        )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset internal state.

        NormalizeTransform is stateless.
        """

        return None


# =============================================================================
# Register Transform
# =============================================================================

TransformRegistry.register(
    "normalize",
    NormalizeTransform,
)