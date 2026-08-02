"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    color_balance.py

Version:
    1.0.0

Description
-----------
Color balancing and color correction transform for BloodCellAI.

Supported Operations
--------------------
✓ Gray World White Balance
✓ Simple White Balance
✓ Gamma Correction
✓ Brightness Adjustment
✓ Contrast Adjustment
✓ Saturation Adjustment
✓ Hue Adjustment

Features
--------
✓ Medical image preprocessing
✓ Color normalization
✓ Illumination correction
✓ Dataset standardization

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
# Color Balance Transform
# =============================================================================

class ColorBalanceTransform(BaseTransform):
    """
    Color balancing transform.

    Supports
    --------
    - White Balance
    - Gamma Correction
    - Brightness
    - Contrast
    - Saturation
    - Hue Adjustment
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__(config)

        self._color_config = config.color

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def parameters(self) -> dict:
        """
        Return transform parameters.
        """

        return {
            "auto_white_balance":
                self._color_config.auto_white_balance,

            "gamma_correction":
                self._color_config.gamma_correction,

            "gamma":
                self._color_config.gamma,

            "brightness":
                self._color_config.brightness,

            "contrast":
                self._color_config.contrast,

            "saturation":
                self._color_config.saturation,

            "hue_shift":
                self._color_config.hue_shift,

            "preserve_color_balance":
                self._color_config.preserve_color_balance,
        }
    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

    

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply color balancing operations.
        """

        result = image.copy()

        # White Balance
        if self._color_config.auto_white_balance:
            result = self._gray_world_white_balance(result)

        # Gamma Correction
        if self._color_config.gamma_correction:
            result = self._gamma_correction(result)

        # Brightness
        if self._color_config.brightness != 0:
            result = self._adjust_brightness(result)

        # Contrast
        if self._color_config.contrast != 1.0:
            result = self._adjust_contrast(result)

        # Saturation
        if self._color_config.saturation != 1.0:
            result = self._adjust_saturation(result)

        # Hue
        if self._color_config.hue_shift != 0:
            result = self._adjust_hue(result)

        return result

    # -------------------------------------------------------------------------
    # Gray World White Balance
    # -------------------------------------------------------------------------

    def _gray_world_white_balance(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Gray World white balancing.
        """

        if image.ndim != 3:
            return image

        image = image.astype(np.float32)

        b, g, r = cv2.split(image)

        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)

        gray = (
            b_mean +
            g_mean +
            r_mean
        ) / 3.0

        b *= gray / max(b_mean, 1e-6)
        g *= gray / max(g_mean, 1e-6)
        r *= gray / max(r_mean, 1e-6)

        balanced = cv2.merge(
            (
                b,
                g,
                r,
            )
        )

        balanced = np.clip(
            balanced,
            0,
            255,
        )

        return balanced.astype(np.uint8)

    # -------------------------------------------------------------------------
    # Simple White Balance
    # -------------------------------------------------------------------------

    def _simple_white_balance(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Stretch each color channel independently.
        """

        if image.ndim != 3:
            return image

        balanced = image.astype(np.float32)

        for channel in range(3):

            minimum = np.min(
                balanced[:, :, channel]
            )

            maximum = np.max(
                balanced[:, :, channel]
            )

            if maximum > minimum:

                balanced[:, :, channel] = (
                    (
                        balanced[:, :, channel]
                        - minimum
                    )
                    * 255.0
                    / (maximum - minimum)
                )

        balanced = np.clip(
            balanced,
            0,
            255,
        )

        return balanced.astype(np.uint8)

    # -------------------------------------------------------------------------
    # Gamma Correction
    # -------------------------------------------------------------------------

    def _gamma_correction(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply gamma correction using a lookup table.
        """

        gamma = self._color_config.gamma

        if gamma <= 0:
            return image

        inverse_gamma = 1.0 / gamma

        table = np.array(
            [
                (
                    (
                        i / 255.0
                    ) ** inverse_gamma
                ) * 255
                for i in range(256)
            ],
            dtype=np.uint8,
        )

        return cv2.LUT(
            image,
            table,
        )

    # -------------------------------------------------------------------------
    # Brightness
    # -------------------------------------------------------------------------

    def _adjust_brightness(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust image brightness.
        """

        image = image.astype(np.float32)

        image += self._color_config.brightness

        image = np.clip(
            image,
            0,
            255,
        )

        return image.astype(np.uint8)

    # -------------------------------------------------------------------------
    # Contrast
    # -------------------------------------------------------------------------

    def _adjust_contrast(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust image contrast.
        """

        image = image.astype(np.float32)

        image = (
            (image - 127.5)
            * self._color_config.contrast
            + 127.5
        )

        image = np.clip(
            image,
            0,
            255,
        )

        return image.astype(np.uint8)
    # -------------------------------------------------------------------------
    # Saturation
    # -------------------------------------------------------------------------

    def _adjust_saturation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust image saturation.
        """

        if image.ndim != 3:
            return image

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        hsv = hsv.astype(np.float32)

        hsv[:, :, 1] *= self._color_config.saturation

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1],
            0,
            255,
        )

        hsv = hsv.astype(np.uint8)

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )
    # -------------------------------------------------------------------------
    # Hue
    # -------------------------------------------------------------------------

    def _adjust_hue(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Shift image hue.
        """

        if image.ndim != 3:
            return image

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        hsv = hsv.astype(np.int16)

        hsv[:, :, 0] += self._color_config.hue_shift

        hsv[:, :, 0] %= 180

        hsv = hsv.astype(np.uint8)

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )
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

        if self._color_config.gamma <= 0:
            raise ValueError(
                "gamma must be greater than zero."
            )

        if self._color_config.contrast <= 0:
            raise ValueError(
                "contrast must be greater than zero."
            )

        if self._color_config.saturation <= 0:
            raise ValueError(
                "saturation must be greater than zero."
            )

    def validate_output(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the processed image.
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
    "color_balance",
    ColorBalanceTransform,
)
