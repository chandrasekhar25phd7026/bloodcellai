"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    denoise.py

Version:
    1.0.0

Description
-----------
Image denoising transform for BloodCellAI.

Supported Methods
-----------------
✓ Gaussian Blur
✓ Median Blur
✓ Bilateral Filter
✓ Non-Local Means

Features
--------
✓ Medical image preprocessing
✓ Edge-preserving filtering
✓ Noise reduction
✓ Configurable parameters

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing_config import PreprocessingConfig
from transforms.base_transform import BaseTransform
from transforms.registry import TransformRegistry


# =============================================================================
# Denoise Transform
# =============================================================================

class DenoiseTransform(BaseTransform):
    """
    Image denoising transform.

    Supports
    --------
    - Gaussian Blur
    - Median Blur
    - Bilateral Filter
    - Non-Local Means
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__(config)

        self._denoise_config = config.denoise

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def parameters(self) -> dict:
        """
        Return transform parameters.
        """

        return {
            "method": self._denoise_config.method,
            "kernel_size": self._denoise_config.kernel_size,
            "sigma": self._denoise_config.sigma,
            "strength": self._denoise_config.strength,
            "preserve_edges": self._denoise_config.preserve_edges,
        }
    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the selected denoising method.
        """

        method = self._denoise_config.method

        if method.lower() == "gaussian":
            return self._gaussian_blur(image)

        if method.lower() == "median":
            return self._median_blur(image)

        if method.lower() == "bilateral":
            return self._bilateral_filter(image)

        if method.lower() in (
            "nlm",
            "non_local_means",
            "non-local-means",
        ):
            return self._non_local_means(image)

        raise ValueError(
            f"Unsupported denoising method: {method}"
        )

    # -------------------------------------------------------------------------
    # Gaussian Blur
    # -------------------------------------------------------------------------

    def _gaussian_blur(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Gaussian blur.
        """

        kernel = self._make_odd(
            self._denoise_config.kernel_size
        )

        return cv2.GaussianBlur(
            image,
            (kernel, kernel),
            self._denoise_config.sigma,
        )

    # -------------------------------------------------------------------------
    # Median Blur
    # -------------------------------------------------------------------------

    def _median_blur(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Median blur.
        """

        kernel = self._make_odd(
            self._denoise_config.kernel_size
        )

        return cv2.medianBlur(
            image,
            kernel,
        )

    # -------------------------------------------------------------------------
    # Bilateral Filter
    # -------------------------------------------------------------------------

    def _bilateral_filter(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Bilateral filtering.
        """

        diameter = self._make_odd(
            self._denoise_config.kernel_size
        )

        sigma = self._denoise_config.strength

        return cv2.bilateralFilter(
            image,
            diameter,
            sigma,
            sigma,
        )

    # -------------------------------------------------------------------------
    # Non-Local Means
    # -------------------------------------------------------------------------

    def _non_local_means(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Non-Local Means denoising.
        """

        strength = float(
            self._denoise_config.strength
        )

        if image.ndim == 2:

            return cv2.fastNlMeansDenoising(
                image,
                None,
                strength,
                7,
                21,
            )

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            strength,
            strength,
            7,
            21,
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _make_odd(
        self,
        value: int,
    ) -> int:
        """
        Ensure kernel size is an odd positive integer.
        """

        value = max(1, int(value))

        if value % 2 == 0:
            value += 1

        return value
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

        if self._denoise_config.kernel_size <= 0:
            raise ValueError(
                "kernel_size must be greater than zero."
            )

        if self._denoise_config.sigma < 0:
            raise ValueError(
                "sigma cannot be negative."
            )

        if self._denoise_config.strength < 0:
            raise ValueError(
                "strength cannot be negative."
            )

    def validate_output(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the denoised image.
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

    def reset(self) -> None:
        """
        Reset internal state.
        """

        self._input_shape = None
        self._input_dtype = None

# =============================================================================
# Register Transform
# =============================================================================

TransformRegistry.register(
    "denoise",
    DenoiseTransform,
)