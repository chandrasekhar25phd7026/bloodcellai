"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    clahe.py

Version:
    1.0.0

Description
-----------
Contrast Limited Adaptive Histogram Equalization (CLAHE)
transform for BloodCellAI.

Features
--------
✓ Grayscale CLAHE
✓ Color CLAHE
✓ LAB color space enhancement
✓ Brightness preservation
✓ Contrast enhancement
✓ Medical image preprocessing

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
# CLAHE Transform
# =============================================================================

class CLAHETransform(BaseTransform):
    """
    Contrast Limited Adaptive Histogram Equalization.

    Supports
    --------
    - Grayscale images
    - RGB/BGR images
    - LAB luminance enhancement
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__(config)

        self._clahe_config = config.clahe

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def parameters(self) -> dict:
        """
        Return transform parameters.
        """

        return {
            "clip_limit": self._clahe_config.clip_limit,
            "tile_grid_size": self._clahe_config.tile_grid_size,
            "apply_to_luminance_only":
                self._clahe_config.apply_to_luminance_only,
            "color_space":
                self._clahe_config.color_space,
            "preserve_brightness":
                self._clahe_config.preserve_brightness,
        }
    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply CLAHE enhancement.
        """

        if image.ndim == 2:
            return self._apply_grayscale(image)

        if self._clahe_config.apply_to_luminance_only:
            return self._apply_luminance(image)

        return self._apply_color(image)

    # -------------------------------------------------------------------------
    # CLAHE Object
    # -------------------------------------------------------------------------

    def _create_clahe(self):
        """
        Create an OpenCV CLAHE object.
        """

        return cv2.createCLAHE(
            clipLimit=self._clahe_config.clip_limit,
            tileGridSize=self._clahe_config.tile_grid_size,
        )

    # -------------------------------------------------------------------------
    # Grayscale CLAHE
    # -------------------------------------------------------------------------

    def _apply_grayscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply CLAHE to a grayscale image.
        """

        clahe = self._create_clahe()

        return clahe.apply(image)

    # -------------------------------------------------------------------------
    # LAB Luminance CLAHE
    # -------------------------------------------------------------------------

    def _apply_luminance(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply CLAHE only to the luminance channel.
        """

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = self._create_clahe()

        enhanced_l = clahe.apply(l_channel)

        merged = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel,
            )
        )

        enhanced = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR,
        )

        if self._clahe_config.preserve_brightness:

            enhanced = self._preserve_brightness(
                image,
                enhanced,
            )

        return enhanced

    # -------------------------------------------------------------------------
    # Full Color CLAHE
    # -------------------------------------------------------------------------

    def _apply_color(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply CLAHE independently to each channel.
        """

        clahe = self._create_clahe()

        channels = cv2.split(image)

        enhanced_channels = []

        for channel in channels:

            enhanced_channels.append(
                clahe.apply(channel)
            )

        enhanced = cv2.merge(
            enhanced_channels
        )

        if self._clahe_config.preserve_brightness:

            enhanced = self._preserve_brightness(
                image,
                enhanced,
            )

        return enhanced

    # -------------------------------------------------------------------------
    # Brightness Preservation
    # -------------------------------------------------------------------------

    def _preserve_brightness(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> np.ndarray:
        """
        Preserve the original image brightness.
        """

        original_mean = np.mean(original)

        enhanced_mean = np.mean(enhanced)

        if enhanced_mean == 0:
            return enhanced

        scale = original_mean / enhanced_mean

        enhanced = enhanced.astype(np.float32)

        enhanced *= scale

        enhanced = np.clip(
            enhanced,
            0,
            255,
        )

        return enhanced.astype(np.uint8)
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

        rows, cols = self._clahe_config.tile_grid_size

        if rows <= 0 or cols <= 0:
            raise ValueError(
                "tile_grid_size values must be greater than zero."
            )

    def validate_output(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the enhanced image.
        """

        super().validate_output(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Output must be a NumPy ndarray."
            )

        if image.dtype != np.uint8:
            raise TypeError(
                "CLAHE output image must have dtype uint8."
            )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset internal state.

        CLAHETransform is stateless.
        """

        return None
# =============================================================================
# Register Transform
# =============================================================================

TransformRegistry.register(
    "clahe",
    CLAHETransform,
)