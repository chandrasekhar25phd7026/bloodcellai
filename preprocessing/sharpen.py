"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    sharpen.py

Version:
    1.0.0

Description
-----------
Image sharpening transform for BloodCellAI.

Supported Methods
-----------------
✓ Unsharp Mask
✓ Sharpen Kernel
✓ Laplacian Sharpening

Features
--------
✓ Medical image preprocessing
✓ Edge enhancement
✓ Detail restoration
✓ Configurable sharpening strength

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
# Sharpen Transform
# =============================================================================

class SharpenTransform(BaseTransform):
    """
    Image sharpening transform.

    Supports
    --------
    - Unsharp Mask
    - Sharpen Kernel
    - Laplacian Sharpening
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__(config)

        self._sharpen_config = config.sharpen

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def parameters(self) -> dict:
        """
        Return transform parameters.
        """

        return {
            "method": self._sharpen_config.method,
            "strength": self._sharpen_config.strength,
            "kernel_size": self._sharpen_config.kernel_size,
            "sigma": self._sharpen_config.sigma,
            "amount": self._sharpen_config.amount,
            "threshold": self._sharpen_config.threshold,
        }
    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the selected sharpening method.
        """

        method = self._sharpen_config.method.lower()

        if method == "unsharp":
            return self._unsharp_mask(image)

        if method == "kernel":
            return self._sharpen_kernel(image)

        if method == "laplacian":
            return self._laplacian_sharpen(image)

        raise ValueError(
            f"Unsupported sharpening method: {method}"
        )

    # -------------------------------------------------------------------------
    # Unsharp Mask
    # -------------------------------------------------------------------------

    def _unsharp_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Unsharp Mask sharpening.
        """

        kernel = self._make_odd(
            self._sharpen_config.kernel_size
        )

        blurred = cv2.GaussianBlur(
            image,
            (kernel, kernel),
            self._sharpen_config.sigma,
        )

        sharpened = cv2.addWeighted(
            image,
            1.0 + self._sharpen_config.amount,
            blurred,
            -self._sharpen_config.amount,
            0,
        )

        if self._sharpen_config.threshold > 0:

            difference = cv2.absdiff(
                image,
                blurred,
            )

            mask = difference > self._sharpen_config.threshold

            sharpened = np.where(
                mask,
                sharpened,
                image,
            )

        return self._clip(sharpened)

    # -------------------------------------------------------------------------
    # Sharpen Kernel
    # -------------------------------------------------------------------------

    def _sharpen_kernel(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply sharpening using a convolution kernel.
        """

        strength = float(
            self._sharpen_config.strength
        )

        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5 + strength, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        sharpened = cv2.filter2D(
            image,
            -1,
            kernel,
        )

        return self._clip(sharpened)

    # -------------------------------------------------------------------------
    # Laplacian Sharpening
    # -------------------------------------------------------------------------

    def _laplacian_sharpen(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Laplacian sharpening.
        """

        if image.ndim == 2:

            laplacian = cv2.Laplacian(
                image,
                cv2.CV_32F,
            )

        else:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            laplacian = cv2.Laplacian(
                gray,
                cv2.CV_32F,
            )

            laplacian = cv2.cvtColor(
                laplacian,
                cv2.COLOR_GRAY2BGR,
            )

        sharpened = (
            image.astype(np.float32)
            - self._sharpen_config.strength
            * laplacian
        )

        return self._clip(sharpened)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _clip(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Clip image values to the valid uint8 range.
        """

        image = np.clip(
            image,
            0,
            255,
        )

        return image.astype(np.uint8)

    def _make_odd(
        self,
        value: int,
    ) -> int:
        """
        Ensure kernel size is an odd positive integer.
        """

        value = max(
            1,
            int(value),
        )

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

        if self._sharpen_config.kernel_size <= 0:
            raise ValueError(
                "kernel_size must be greater than zero."
            )

        if self._sharpen_config.sigma < 0:
            raise ValueError(
                "sigma cannot be negative."
            )

        if self._sharpen_config.amount < 0:
            raise ValueError(
                "amount cannot be negative."
            )

        if self._sharpen_config.strength < 0:
            raise ValueError(
                "strength cannot be negative."
            )

        if self._sharpen_config.threshold < 0:
            raise ValueError(
                "threshold cannot be negative."
            )

    def validate_output(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the sharpened image.
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
    "sharpen",
    SharpenTransform,
)