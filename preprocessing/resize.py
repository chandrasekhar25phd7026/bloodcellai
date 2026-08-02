"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    resize.py

Version:
    1.0.0

Description
-----------
Image resizing transform for the BloodCellAI preprocessing framework.

Features
--------
✓ Standard image resizing
✓ Aspect ratio preservation
✓ Letterbox padding
✓ Multiple interpolation methods
✓ Compatible with OpenCV
✓ Compatible with YOLO preprocessing

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing_config import PreprocessingConfig
from .preprocessing_models import InterpolationMethod
from transforms.base_transform import BaseTransform
from transforms.registry import TransformRegistry


# =============================================================================
# OpenCV Interpolation Mapping
# =============================================================================

_INTERPOLATION_MAP = {
    InterpolationMethod.NEAREST: cv2.INTER_NEAREST,
    InterpolationMethod.LINEAR: cv2.INTER_LINEAR,
    InterpolationMethod.CUBIC: cv2.INTER_CUBIC,
    InterpolationMethod.AREA: cv2.INTER_AREA,
    InterpolationMethod.LANCZOS: cv2.INTER_LANCZOS4,
}


# =============================================================================
# Resize Transform
# =============================================================================

class ResizeTransform(BaseTransform):
    """
    Resize images according to the preprocessing configuration.

    Supports
    --------
    - Standard resize
    - Aspect-ratio preserving resize
    - Letterbox resize
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        super().__init__(config)

        self._resize_config = config.resize

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def parameters(self) -> dict:
        """
        Return transform parameters.
        """

        return {
            "target_width": self._resize_config.target_width,
            "target_height": self._resize_config.target_height,
            "keep_aspect_ratio": self._resize_config.keep_aspect_ratio,
            "pad_image": self._resize_config.pad_image,
            "padding_value": self._resize_config.padding_value,
            "interpolation": self._resize_config.interpolation.value,
        }
    # -------------------------------------------------------------------------
    # Apply
    # -------------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply image resizing.
        """

        if self._resize_config.keep_aspect_ratio:

            return self._letterbox(image)

        return self._resize(image)

    # -------------------------------------------------------------------------
    # Standard Resize
    # -------------------------------------------------------------------------

    def _resize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize image without preserving aspect ratio.
        """

        interpolation = _INTERPOLATION_MAP[
            self._resize_config.interpolation
        ]

        return cv2.resize(
            image,
            (
                self._resize_config.target_width,
                self._resize_config.target_height,
            ),
            interpolation=interpolation,
        )

    # -------------------------------------------------------------------------
    # Letterbox Resize
    # -------------------------------------------------------------------------

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize while preserving aspect ratio and optionally pad.
        """

        original_height, original_width = image.shape[:2]

        target_width = self._resize_config.target_width
        target_height = self._resize_config.target_height

        scale = min(
            target_width / original_width,
            target_height / original_height,
        )

        new_width = int(round(original_width * scale))
        new_height = int(round(original_height * scale))

        interpolation = _INTERPOLATION_MAP[
            self._resize_config.interpolation
        ]

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )

        if not self._resize_config.pad_image:
            return resized

        if image.ndim == 2:

            canvas = np.full(
                (
                    target_height,
                    target_width,
                ),
                self._resize_config.padding_value,
                dtype=image.dtype,
            )

        else:

            canvas = np.full(
                (
                    target_height,
                    target_width,
                    image.shape[2],
                ),
                self._resize_config.padding_value,
                dtype=image.dtype,
            )

        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width,
        ] = resized

        return canvas
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
        Validate the resized image.
        """

        super().validate_output(image)

        expected_height = self._resize_config.target_height
        expected_width = self._resize_config.target_width

        if self._resize_config.keep_aspect_ratio:

            if self._resize_config.pad_image:

                if image.shape[0] != expected_height:
                    raise ValueError(
                        "Invalid output height."
                    )

                if image.shape[1] != expected_width:
                    raise ValueError(
                        "Invalid output width."
                    )

        else:

            if image.shape[0] != expected_height:
                raise ValueError(
                    "Invalid output height."
                )

            if image.shape[1] != expected_width:
                raise ValueError(
                    "Invalid output width."
                )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset internal state.

        ResizeTransform is stateless, so nothing needs
        to be reset.
        """

        return None

# =============================================================================
# Register Transform
# =============================================================================

TransformRegistry.register(
    "resize",
    ResizeTransform,
)
