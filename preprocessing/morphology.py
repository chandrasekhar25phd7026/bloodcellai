"""
BloodCellAI
Morphological Image Processing Transform

Provides classical morphological operations for
medical image preprocessing.

Supported Operations

- Erosion
- Dilation
- Opening
- Closing
- Morphological Gradient
- Top Hat
- Black Hat
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
class MorphologyOperation(
    Enum,
):
    """
    Supported morphological operations.
    """

    EROSION = "erosion"

    DILATION = "dilation"

    OPENING = "opening"

    CLOSING = "closing"

    GRADIENT = "gradient"

    TOP_HAT = "top_hat"

    BLACK_HAT = "black_hat"

class MorphologyTransform(
    BaseTransform,
):
    """
    Morphological preprocessing transform.

    Supports several OpenCV morphology operations
    using configurable structuring elements.
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

        morphology = getattr(
            config,
            "morphology",
            None,
        )

        if morphology is None:

            raise ValueError(
                "Morphology configuration "
                "is missing."
            )

        self._enabled = morphology.enabled

        self._operation = MorphologyOperation(
            morphology.operation
        )

        self._kernel_size = (
            morphology.kernel_size
        )

        self._iterations = (
            morphology.iterations
        )

        self._kernel_shape = (
            morphology.kernel_shape
        )

        self._kernel = None

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
                "Empty image."
            )

        if self._kernel_size <= 0:

            raise ValueError(
                "Kernel size must "
                "be greater than zero."
            )

        if self._iterations <= 0:

            raise ValueError(
                "Iterations must "
                "be greater than zero."
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

            "kernel_size":
                self._kernel_size,

            "iterations":
                self._iterations,

            "kernel_shape":
                self._kernel_shape,

        }

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

            "kernel_size":
                self._kernel_size,

            "iterations":
                self._iterations,

            "kernel_shape":
                self._kernel_shape,

            "kernel":
                None
                if self._kernel is None
                else self._kernel.tolist(),

        }

    # ---------------------------------------------------------
    # Kernel Creation
    # ---------------------------------------------------------

    def _create_kernel(
        self,
    ) -> np.ndarray:
        """
        Create the structuring element used for
        morphological operations.
        """

        shape = self._kernel_shape.lower()

        if shape == "rectangle":

            kernel_type = cv2.MORPH_RECT

        elif shape == "ellipse":

            kernel_type = cv2.MORPH_ELLIPSE

        elif shape == "cross":

            kernel_type = cv2.MORPH_CROSS

        else:

            raise ValueError(

                f"Unsupported kernel shape: "

                f"{self._kernel_shape}"

            )

        kernel = cv2.getStructuringElement(

            kernel_type,

            (

                self._kernel_size,

                self._kernel_size,

            ),

        )

        self._kernel = kernel

        return kernel

    # ---------------------------------------------------------
    # Kernel Creation
    # ---------------------------------------------------------

    def _create_kernel(
        self,
    ) -> np.ndarray:
        """
        Create the structuring element used for
        morphological operations.
        """

        shape = self._kernel_shape.lower()

        if shape == "rectangle":

            kernel_type = cv2.MORPH_RECT

        elif shape == "ellipse":

            kernel_type = cv2.MORPH_ELLIPSE

        elif shape == "cross":

            kernel_type = cv2.MORPH_CROSS

        else:

            raise ValueError(

                f"Unsupported kernel shape: "

                f"{self._kernel_shape}"

            )

        kernel = cv2.getStructuringElement(

            kernel_type,

            (

                self._kernel_size,

                self._kernel_size,

            ),

        )

        self._kernel = kernel

        return kernel

    # ---------------------------------------------------------
    # Kernel Getter
    # ---------------------------------------------------------

    def _get_kernel(
        self,
    ) -> np.ndarray:
        """
        Return the cached kernel.
        """

        if self._kernel is None:

            return self._create_kernel()

        return self._kernel

    # ---------------------------------------------------------
    # Kernel Information
    # ---------------------------------------------------------

    @property
    def kernel(
        self,
    ) -> np.ndarray:
        """
        Return current kernel.
        """

        return self._get_kernel()

    @property
    def kernel_size(
        self,
    ) -> int:

        return self._kernel_size

    @property
    def operation(
        self,
    ) -> MorphologyOperation:

        return self._operation

    @property
    def operation(
        self,
    ) -> MorphologyOperation:

        return self._operation

    @property
    def kernel_shape(
        self,
    ) -> str:

        return self._kernel_shape


    # ---------------------------------------------------------
    # Erosion
    # ---------------------------------------------------------

    def _erosion(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply erosion.
        """

        kernel = self._get_kernel()

        return cv2.erode(

            image,

            kernel,

            iterations=self._iterations,

        )

    # ---------------------------------------------------------
    # Dilation
    # ---------------------------------------------------------

    def _dilation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply dilation.
        """

        kernel = self._get_kernel()

        return cv2.dilate(

            image,

            kernel,

            iterations=self._iterations,

        )

    # ---------------------------------------------------------
    # Dilation
    # ---------------------------------------------------------

    def _dilation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply dilation.
        """

        kernel = self._get_kernel()

        return cv2.dilate(

            image,

            kernel,

            iterations=self._iterations,

        )

    # ---------------------------------------------------------
    # Closing
    # ---------------------------------------------------------

    def _closing(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply morphological closing.
        """

        kernel = self._get_kernel()

        return cv2.morphologyEx(

            image,

            cv2.MORPH_CLOSE,

            kernel,

            iterations=self._iterations,

        )

    # ---------------------------------------------------------
    # Closing
    # ---------------------------------------------------------

    def _closing(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply morphological closing.
        """

        kernel = self._get_kernel()

        return cv2.morphologyEx(

            image,

            cv2.MORPH_CLOSE,

            kernel,

            iterations=self._iterations,

        )

    # ---------------------------------------------------------
    # Basic Morphology Dispatcher
    # ---------------------------------------------------------

    def _apply_basic_operation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply one of the basic morphology
        operations.
        """

        if self._operation == MorphologyOperation.EROSION:

            return self._erosion(
                image,
            )

        if self._operation == MorphologyOperation.DILATION:

            return self._dilation(
                image,
            )

        if self._operation == MorphologyOperation.OPENING:

            return self._opening(
                image,
            )

        if self._operation == MorphologyOperation.CLOSING:

            return self._closing(
                image,
            )

        raise ValueError(

            f"Unsupported operation: "

            f"{self._operation.value}"

        )

    # ---------------------------------------------------------
    # Basic Morphology Dispatcher
    # ---------------------------------------------------------

    def _apply_basic_operation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply one of the basic morphology
        operations.
        """

        if self._operation == MorphologyOperation.EROSION:

            return self._erosion(
                image,
            )

        if self._operation == MorphologyOperation.DILATION:

            return self._dilation(
                image,
            )

        if self._operation == MorphologyOperation.OPENING:

            return self._opening(
                image,
            )

        if self._operation == MorphologyOperation.CLOSING:

            return self._closing(
                image,
            )

        raise ValueError(

            f"Unsupported operation: "

            f"{self._operation.value}"

        )

    # ---------------------------------------------------------
    # Morphological Gradient
    # ---------------------------------------------------------

    def _gradient(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply morphological gradient.
        """

        kernel = self._get_kernel()

        return cv2.morphologyEx(

            image,

            cv2.MORPH_GRADIENT,

            kernel,

            iterations=self._iterations,

        )

    # ---------------------------------------------------------
    # Black Hat
    # ---------------------------------------------------------

    def _black_hat(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Black Hat transformation.
        """

        kernel = self._get_kernel()

        return cv2.morphologyEx(

            image,

            cv2.MORPH_BLACKHAT,

            kernel,

            iterations=self._iterations,

        )

    # ---------------------------------------------------------
    # Dispatcher
    # ---------------------------------------------------------

    def _apply_operation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the configured morphology
        operation.
        """

        if self._operation in (

            MorphologyOperation.EROSION,

            MorphologyOperation.DILATION,

            MorphologyOperation.OPENING,

            MorphologyOperation.CLOSING,

        ):

            return self._apply_basic_operation(
                image
            )

        if self._operation == MorphologyOperation.GRADIENT:

            return self._gradient(
                image
            )

        if self._operation == MorphologyOperation.TOP_HAT:

            return self._top_hat(
                image
            )

        if self._operation == MorphologyOperation.BLACK_HAT:

            return self._black_hat(
                image
            )

        raise ValueError(

            f"Unsupported morphology "

            f"operation: "

            f"{self._operation.value}"

        )

    # ---------------------------------------------------------
    # Before Apply
    # ---------------------------------------------------------

    def before_apply(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Executed before morphology.
        """

        self.validate_input(
            image
        )

        self._logger.debug(

            "Applying %s",

            self._operation.value,

        )

    # ---------------------------------------------------------
    # Apply
    # ---------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the configured
        morphological operation.
        """

        if not self._enabled:

            return image

        return self._apply_operation(
            image
        )

    # ---------------------------------------------------------
    # Validate Output
    # ---------------------------------------------------------

    def validate_output(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate processed image.
        """

        if image is None:

            raise ValueError(
                "Morphology produced "
                "a None image."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):

            raise TypeError(
                "Output must be "
                "numpy.ndarray."
            )

        if image.size == 0:

            raise ValueError(
                "Output image is empty."
            )

    # ---------------------------------------------------------
    # After Apply
    # ---------------------------------------------------------

    def after_apply(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Executed after processing.
        """

        self.validate_output(
            image
        )

        self._logger.debug(

            "Morphology completed."

        )

        def __call__(
        self,
        image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        TransformRecord,
    ]:

        try:

            self.before_apply(
                image
            )

            output = self.apply(
                image
            )

            self.after_apply(
                output
            )

            record = TransformRecord(

                name="Morphology",

                status=TransformStatus.SUCCESS,

                parameters=self.parameters,

                metadata={

                    "operation":
                        self._operation.value,

                    "kernel_size":
                        self._kernel_size,

                    "iterations":
                        self._iterations,

                    "kernel_shape":
                        self._kernel_shape,

                },

            )

            return (

                output,

                record,

            )

        except Exception as error:

            self._logger.exception(

                "Morphology failed."

            )

            record = TransformRecord(

                name="Morphology",

                status=TransformStatus.FAILED,

                parameters=self.parameters,

                metadata={

                    "error": str(error),

                },

            )

            raise

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        return (

            f"{self.__class__.__name__}("

            f"operation={self._operation.value}, "

            f"kernel_size={self._kernel_size}, "

            f"iterations={self._iterations}, "

            f"kernel_shape='{self._kernel_shape}', "

            f"enabled={self._enabled}"

            f")"

        )

    def __str__(
        self,
    ) -> str:

        return (

            f"MorphologyTransform("

            f"{self._operation.value}"

            f")"

        )
    # ---------------------------------------------------------
    # Enable / Disable
    # ---------------------------------------------------------

    def enable(
        self,
    ) -> None:

        self._enabled = True


    def disable(
        self,
    ) -> None:

        self._enabled = False

    @property
    def enabled(
        self,
    ) -> bool:

        return self._enabled
    @property
    def enabled(
        self,
    ) -> bool:

        return self._enabled


# ---------------------------------------------------------
# Register Transform
# ---------------------------------------------------------

TransformRegistry.register(

    "morphology",

    MorphologyTransform,

)